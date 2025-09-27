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
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


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
        self.prompts_dir = None

        # Initialize Anthropic client
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.sonnet_model = "claude-sonnet-4-20250514"
        self.opus_model = "claude-opus-4-1-20250805"

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
                data = json.load(f)

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

    def format_my_team(
        self, my_team_data: dict[str, Any], free_transfers: int = 1
    ) -> str:
        """Format my team data into a readable description using LLM."""
        try:
            current_picks = my_team_data.get("current_picks", [])
            summary = my_team_data.get("summary", {})
            team_value_info = my_team_data.get("team_value", {})

            # Prepare team data for LLM formatting
            team_context = {
                "team_name": summary.get("team_name", "Unknown"),
                "total_points": summary.get("total_points", 0),
                "overall_rank": summary.get("overall_rank", 0),
                "gameweek_points": summary.get("gameweek_points", 0),
                "team_value": team_value_info.get("team_value", 0),
                "bank_balance": team_value_info.get("bank_balance", 0),
                "free_transfers": free_transfers,
                "current_picks": current_picks,
            }

            prompt = f"""Format this FPL team data into a clear, readable summary:

{json.dumps(team_context, indent=2)}

Create a concise summary including:
- Team name and current performance
- Starting XI with positions and key stats
- Bench players
- Team value and bank balance
- Free transfers available
- Current captain/vice-captain

Format it as clear prose, not JSON."""

            response = self._make_anthropic_call(
                model=self.sonnet_model,
                prompt=prompt,
                system="You are an FPL analyst. Format team data clearly and concisely.",
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
        self, model: str, prompt: str, system: str, max_tokens: int = 4000
    ) -> str:
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

            response_text = message.content[0].text.strip()
            self.logger.debug(
                f"API call successful, response length: {len(response_text)}"
            )
            return response_text

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
        try:
            channel_name = channel_data.get("channel_name", "Unknown")
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

TOP FPL PLAYERS BY OWNERSHIP (showing first 75 for context - includes injury status):
{json.dumps(condensed_players[:75], indent=1)}

PLAYER STATUS CODES:
- a: available, d: doubtful, i: injured, s: suspended, u: unavailable
- Pay attention to "status", "news", and "chance_of_playing_next_round" fields

TRANSCRIPT:
{transcript}

Extract the following information and return as JSON:

{{
  "channel_name": "{channel_name}",
  "formation": "3-5-2",
  "team_selection": ["Salah (FWD)", "Haaland (FWD)", ...],
  "transfers_in": ["Player (POS)", ...],
  "transfers_out": ["Player (POS)", ...], 
  "captain_choice": "Player (POS)",
  "vice_captain_choice": "Player (POS)",
  "key_issues_discussed": [
    {{"issue": "Salah vs Haaland captaincy", "opinion": "On pens, great fixtures in GW5"}},
    {{"issue": "Arsenal defensive assets", "opinion": "Avoid due to tough fixtures next 3 weeks"}}
  ],
  "watchlist": [
    {{"name": "Player (POS)", "priority": "high", "why": "Great fixtures coming up after international break"}}
  ],
  "bank_itb": "0.5m",
  "key_reasoning": ["Reason 1", "Reason 2", ...],
  "confidence": 0.85,
  "transcript_length": {transcript_length}
}}

IMPORTANT:
- The transcript may have transcription errors for player names
- Use the top players list to identify correct player names and their injury status
- ALL PLAYER NAMES must include position in format: "Player (POS)" e.g. "Salah (FWD)", "Robertson (DEF)"
- Extract their actual team selection, transfers, and reasoning
- Formation: Look for tactical discussions (3-5-2, 4-4-2, etc.) - set null if not mentioned
- Key Issues: Extract major talking points with their specific opinions (if any discussed)
- Watchlist: Players they mention considering but not immediately transferring (high/med/low priority)
- Bank ITB: If they mention money in the bank or ITB, capture the amount (e.g. "0.5m", "2.1m")
- Consider player availability (status/news/chance_of_playing_next_round) in analysis
- Set confidence based on clarity of their decisions and transcript length
- If information is unclear or missing, use empty arrays for lists and default values
- FOR SHORT TRANSCRIPTS (<3000 chars): Focus on extracting the most essential information (transfers, captain)
- ALWAYS return valid JSON even if limited information is available
"""

            system = """You are an expert FPL analyst. Extract structured information from influencer video transcripts.
Focus on concrete decisions: team selections, transfers, captain choices, and key reasoning.
Return valid JSON only. For short transcripts, extract whatever information is available."""

            # Save prompt if debug mode is on
            self.save_debug_content(f"{channel_name}_prompt.txt", prompt)

            response = self._make_anthropic_call(
                model=self.sonnet_model, prompt=prompt, system=system, max_tokens=2000
            )

            # Save response if debug mode is on
            self.save_debug_content(f"{channel_name}_response.json", response)

            # Parse JSON response
            try:
                # Clean up response if it has code fences
                if "```" in response:
                    import re

                    json_match = re.search(
                        r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL
                    )
                    if json_match:
                        response = json_match.group(1)

                analysis_data = json.loads(response)

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
                    f"JSON parsing failed for {channel_name}: {e}\nResponse: {response[:500]}..."
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
                self.logger.error(
                    f"Validation failed for {channel_name}: {e}\nData: {json.dumps(analysis_data, indent=2)[:500]}..."
                )
                return None

        except Exception as e:
            self.logger.error(f"Error analyzing channel {channel_name}: {e}")
            return None

    def generate_comparative_analysis(
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

STYLE
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
- my_team: [squad[15] with positions/teams/prices/sell_prices, starting_xi?, bench_order?, captain?, vice?, itb, free_transfers (IMPORTANT: consider this number (and the number of hits beyond that number) when making transfer recommendations), planned_hits?, team_value, chips_available, risk_tolerance?]
- context (optional): [gw, deadline_utc, fixture_difficulty, minutes/injury flags, projections, blank/double indicators, price_change_risk]

FPL VALIDATION RULES FOR ALL RECOMMENDATIONS
- Obey budget (use sell_prices if provided; otherwise use current prices and label as estimate).
- Max 3 players per real club.
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
- Key issues discussed (their main talking points)
- Chip talk (if any)
Keep this section concise; use bullets. Always cite this influencer after their opinions.

## 3) My Team vs Influencers (Gap Analysis)
- **Ensure you list every single player in my starting XI if known (with position), and bench**
- **Players I have that are being benched/sold by influencers**: list with citations.
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
  - Impact of taking hits: (extra transfers beyond free) × 4 points.
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

### Captain & Vice
- Recommend captain and vice, with Safe vs Upside tags and citations.

### Starting XI & Bench Order
- List recommended XI by position.
- Show bench order (1/2/3) and highlight any 50/50 calls, with reasoning.

### Chips
- Only recommend chip use if influencers strongly advocate or if gameweek context demands it (e.g., doubles/blanks). Otherwise: “No chip recommended.” Cite any mentions.

### Future Planning (GW+1 to GW+3)
- Suggest 2-4 priority future targets with reasoning (fixtures, form, price).
- Recommend when to roll to set up a bigger move.
- Call out potential early vs late transfer timing (e.g., due to price rises).

--------------------------------------------------
CRITICAL REQUIREMENTS
- {critical_directive_line}
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

            response = self._make_anthropic_call(
                model=self.opus_model, prompt=prompt, system=system, max_tokens=6000
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
                if channel_name and channel_name in transcripts:
                    transcript_data = transcripts[channel_name]
                    transcript = transcript_data.get("transcript", "")
                    analysis = self.analyze_channel(
                        video_data, condensed_players, transcript, gameweek
                    )
                    if analysis:
                        channel_analyses.append(analysis)
                else:
                    self.logger.warning(
                        f"No transcript found for {channel_name or 'Unknown'}"
                    )

            if not channel_analyses:
                raise ValueError("No successful channel analyses - cannot proceed")

            self.logger.info(f"Successfully analyzed {len(channel_analyses)} channels")

            # Phase 2: Format my team and generate comparative analysis
            self.logger.info("Phase 2: Generating comparative analysis")
            self.logger.info(f"Free transfers available: {free_transfers}")
            my_team_summary = self.format_my_team(my_team_data, free_transfers)

            final_report = self.generate_comparative_analysis(
                channel_analyses,
                condensed_players,
                my_team_summary,
                gameweek,
                commentary=commentary,
            )

            # Add header with metadata
            directive_line = ""
            if commentary and commentary.strip():
                directive_line = f"**User Directive:** {commentary.strip()}\n\n"

            report_header = (
                f"# FPL Intelligence Analysis - Gameweek {gameweek}\n\n"
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
                f"**Channels Analyzed:** {len(channel_analyses)}  \n"
                f"**Data Source:** {Path(input_file).name}\n\n"
                f"{directive_line}"
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
