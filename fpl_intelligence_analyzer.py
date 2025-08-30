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
    formation: str | None
    team_selection: list[str]
    transfers_in: list[str]
    transfers_out: list[str]
    captain_choice: str
    vice_captain_choice: str
    key_issues_discussed: list[dict[str, str]]
    watchlist: list[dict[str, str]]
    bank_itb: str | None
    key_reasoning: list[str]
    confidence: float


class FPLIntelligenceAnalyzer:
    """Main class for FPL intelligence analysis using LLM calls."""

    def __init__(self, verbose: bool = False):
        """Initialize the analyzer with logging and Anthropic client."""
        self.setup_logging(verbose)
        self.logger = logging.getLogger(__name__)

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

    def format_my_team(self, my_team_data: dict[str, Any]) -> str:
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
                "current_picks": current_picks,
            }

            prompt = f"""Format this FPL team data into a clear, readable summary:

{json.dumps(team_context, indent=2)}

Create a concise summary including:
- Team name and current performance
- Starting XI with positions and key stats
- Bench players
- Team value and bank balance
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

            self.logger.info(f"Analyzing channel: {channel_name}")

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
  "confidence": 0.85
}}

IMPORTANT:
- The transcript may have transcription errors for player names
- Use the top players list to identify correct player names and their injury status
- ALL PLAYER NAMES must include position in format: "Player (POS)" e.g. "Salah (FWD)", "Robertson (DEF)"
- Extract their actual team selection, transfers, and reasoning
- Formation: Look for tactical discussions (3-5-2, 4-4-2, etc.) - set null if not mentioned
- Key Issues: Extract 4-5 major talking points with their specific opinions on each
- Watchlist: Players they mention considering but not immediately transferring (high/med/low priority)
- Bank ITB: If they mention money in the bank or ITB, capture the amount (e.g. "0.5m", "2.1m")
- Consider player availability (status/news/chance_of_playing_next_round) in analysis
- Set confidence based on clarity of their decisions
- If information is unclear or missing, use null for optional fields or empty arrays for lists
"""

            system = """You are an expert FPL analyst. Extract structured information from influencer video transcripts.
Focus on concrete decisions: team selections, transfers, captain choices, and key reasoning.
Return valid JSON only."""

            response = self._make_anthropic_call(
                model=self.sonnet_model, prompt=prompt, system=system, max_tokens=2000
            )

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
                analysis = ChannelAnalysis(**analysis_data)

                self.logger.info(
                    f"Successfully analyzed {channel_name} (confidence: {analysis.confidence})"
                )
                return analysis

            except (json.JSONDecodeError, ValidationError) as e:
                self.logger.error(
                    f"Failed to parse channel analysis for {channel_name}: {e}"
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
                    issues_list = [f"'{issue['issue']}': {issue['opinion']}" for issue in analysis.key_issues_discussed]
                    key_issues = f"\n- Key Issues: {'; '.join(issues_list)}"
                
                # Format watchlist
                watchlist = ""
                if analysis.watchlist:
                    watch_items = [f"{item['name']} ({item['priority']}: {item['why']})" for item in analysis.watchlist]
                    watchlist = f"\n- Watchlist: {'; '.join(watch_items)}"
                
                # Format formation and bank
                formation = f"\n- Formation: {analysis.formation}" if analysis.formation else ""
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

            # Create comprehensive prompt for Opus-4
            prompt = f"""Generate a comprehensive FPL analysis report for gameweek {gameweek}.

MY CURRENT TEAM:
{my_team_summary}

TOP PLAYERS REFERENCE (first 75 - includes injury/availability data):
{json.dumps(condensed_players[:75], indent=1)}

PLAYER STATUS CODES:
- a: available, d: doubtful, i: injured, s: suspended, u: unavailable
- Consider "status", "news", and "chance_of_playing_next_round" in all recommendations

INFLUENCER ANALYSES:
{combined_summaries}

Create a detailed markdown report with the following structure:

## 1. Executive Summary
- Key consensus picks and major disagreements across all influencers
- Most important decisions for this gameweek
- Overall confidence level and reliability of recommendations

## 2. Channel-by-Channel Breakdown
For each influencer, provide a structured analysis including:
- Formation strategy (if mentioned)
- Key transfers and specific reasoning
- Captain choice with rationale
- Major issues/topics they discussed
- Watchlist players and priorities
- Confidence assessment

## 3. My Team vs Influencers Comparison
Direct comparison showing:
- Players I currently have that influencers are benching/dropping (cite specific channels)
- Popular picks across influencers that I'm missing
- Formation differences if applicable
- Starting XI differences and bench strategies

## 4. Transfer Analysis by Scenario
Provide specific options based on transfer availability:
- **If you have 1 free transfer**: Priority moves with reasoning
- **If you have 2+ free transfers**: Multi-transfer strategies 
- **If considering a hit (-4 points)**: High-confidence moves only
- Always specify which influencers support each recommendation

## 5. Captain Analysis
Detailed breakdown with proper citations:
- **Consensus captain picks**: Players backed by multiple influencers (specify which ones)
- **Split opinions**: Different captain choices with specific reasoning from each channel
- **Risk vs Reward**: Safe vs differential captain options
- **Contrarian picks**: Unique captain choices and why

## 6. Consensus vs Contrarian Analysis
Clear identification with specific attributions:
- **Universal picks**: Players ALL analyzed influencers are backing
- **Majority consensus**: Players backed by most (specify exact channels)
- **Split decisions**: 50/50 or varied opinions with citations
- **Unique differentials**: Picks only mentioned by specific channels

## 7. Watchlist & Future Planning
Compiled from all influencer watchlists:
- **High priority targets**: Players mentioned by multiple channels
- **Medium/Low priority**: Secondary considerations
- **Formation trends**: Tactical shifts being considered
- **Banking strategy**: ITB recommendations and timing

## 8. Conditional Action Plan
Provide multiple strategic options:
- **Primary recommendation**: Most consensus-backed strategy with specific steps
- **Alternative strategy**: For different risk tolerances or situations
- **Timeline considerations**: This week priorities vs future planning
- **Injury contingencies**: Backup plans if key players become unavailable

CRITICAL REQUIREMENTS:
- Always cite specific influencers for opinions: "(FPL Harry)", "(Let's Talk FPL, FPL Raptor)"
- Factor in player injury status and availability for ALL recommendations
- Provide conditional advice rather than assuming transfer count
- Use the structured data (formations, key_issues_discussed, watchlists) effectively
- Make recommendations actionable and specific to my current team situation

Return the report in clean markdown format."""

            system = """You are an expert FPL strategist and analyst. Generate comprehensive, actionable FPL advice 
by analyzing multiple influencer perspectives alongside detailed player data. Your recommendations should be 
specific, well-reasoned, and tailored to the user's current team situation."""

            response = self._make_anthropic_call(
                model=self.opus_model, prompt=prompt, system=system, max_tokens=6000
            )

            self.logger.info("Comparative analysis generated successfully")
            return response

        except Exception as e:
            self.logger.error(f"Error generating comparative analysis: {e}")
            return f"# Analysis Error\n\nFailed to generate comparative analysis: {e!s}"

    def run_analysis(self, input_file: str, output_file: str | None = None) -> None:
        """Run the complete FPL intelligence analysis pipeline."""
        try:
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
                video_id = video_data.get("video_id")
                if video_id and video_id in transcripts:
                    transcript = transcripts[video_id]
                    analysis = self.analyze_channel(
                        video_data, condensed_players, transcript, gameweek
                    )
                    if analysis:
                        channel_analyses.append(analysis)
                else:
                    self.logger.warning(
                        f"No transcript found for {video_data.get('channel_name', 'Unknown')}"
                    )

            if not channel_analyses:
                raise ValueError("No successful channel analyses - cannot proceed")

            self.logger.info(f"Successfully analyzed {len(channel_analyses)} channels")

            # Phase 2: Format my team and generate comparative analysis
            self.logger.info("Phase 2: Generating comparative analysis")
            my_team_summary = self.format_my_team(my_team_data)

            final_report = self.generate_comparative_analysis(
                channel_analyses, condensed_players, my_team_summary, gameweek
            )

            # Add header with metadata
            report_header = f"""# FPL Intelligence Analysis - Gameweek {gameweek}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Channels Analyzed:** {len(channel_analyses)}  
**Data Source:** {Path(input_file).name}

---

"""

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

    args = parser.parse_args()

    try:
        analyzer = FPLIntelligenceAnalyzer(verbose=args.verbose)
        analyzer.run_analysis(args.input, args.output_file)
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
