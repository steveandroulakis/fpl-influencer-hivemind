"""Tests for orchestrator commentary-driven option generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from src.fpl_influencer_hivemind.analyzer.models import OptionRequest

if TYPE_CHECKING:
    from pytest import MonkeyPatch


class TestParseOptionRequests:
    """Tests for _parse_option_requests method."""

    @pytest.fixture
    def mock_analyzer(self, monkeypatch: MonkeyPatch) -> Any:
        """Create analyzer with mocked Anthropic client."""
        # Patch the API key check
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from src.fpl_influencer_hivemind.analyzer.orchestrator import (
            FPLIntelligenceAnalyzer,
        )

        analyzer = FPLIntelligenceAnalyzer(verbose=False, save_prompts=False)
        return analyzer

    def test_returns_empty_for_none_commentary(self, mock_analyzer: Any) -> None:
        """Should return empty list when commentary is None."""
        result = mock_analyzer._parse_option_requests(None)
        assert result == []

    def test_returns_empty_for_empty_string(self, mock_analyzer: Any) -> None:
        """Should return empty list for empty string."""
        result = mock_analyzer._parse_option_requests("")
        assert result == []

    def test_parses_single_transfer_request(self, mock_analyzer: Any) -> None:
        """Should parse a single transfer request."""
        with patch.object(
            mock_analyzer.client,
            "call_haiku",
            return_value=('[{"transfer_count": 1, "take_hit": false}]', "end_turn"),
        ):
            result = mock_analyzer._parse_option_requests("Give me 1 transfer option")

        assert len(result) == 1
        assert result[0].transfer_count == 1
        assert result[0].take_hit is False

    def test_parses_multiple_transfer_requests(self, mock_analyzer: Any) -> None:
        """Should parse multiple transfer requests."""
        with patch.object(
            mock_analyzer.client,
            "call_haiku",
            return_value=(
                '[{"transfer_count": 1, "take_hit": false}, {"transfer_count": 2, "take_hit": true}]',
                "end_turn",
            ),
        ):
            result = mock_analyzer._parse_option_requests(
                "Show me 1 transfer and 2 transfers with hit"
            )

        assert len(result) == 2
        assert result[0].transfer_count == 1
        assert result[0].take_hit is False
        assert result[1].transfer_count == 2
        assert result[1].take_hit is True

    def test_deduplicates_requests(self, mock_analyzer: Any) -> None:
        """Should deduplicate identical requests."""
        with patch.object(
            mock_analyzer.client,
            "call_haiku",
            return_value=(
                '[{"transfer_count": 1, "take_hit": false}, {"transfer_count": 1, "take_hit": false}]',
                "end_turn",
            ),
        ):
            result = mock_analyzer._parse_option_requests("1 transfer, 1 transfer")

        assert len(result) == 1

    def test_handles_json_in_code_fence(self, mock_analyzer: Any) -> None:
        """Should handle JSON wrapped in code fence."""
        with patch.object(
            mock_analyzer.client,
            "call_haiku",
            return_value=(
                '```json\n[{"transfer_count": 0, "take_hit": false}]\n```',
                "end_turn",
            ),
        ):
            result = mock_analyzer._parse_option_requests("Roll the transfer")

        assert len(result) == 1
        assert result[0].transfer_count == 0

    def test_returns_empty_on_parse_error(self, mock_analyzer: Any) -> None:
        """Should return empty list when JSON parsing fails."""
        with patch.object(
            mock_analyzer.client,
            "call_haiku",
            return_value=("not valid json", "end_turn"),
        ):
            result = mock_analyzer._parse_option_requests("invalid input")

        assert result == []

    def test_returns_empty_on_api_error(self, mock_analyzer: Any) -> None:
        """Should return empty list when API call fails."""
        with patch.object(
            mock_analyzer.client,
            "call_haiku",
            side_effect=Exception("API error"),
        ):
            result = mock_analyzer._parse_option_requests("some commentary")

        assert result == []


class TestBuildDynamicLabel:
    """Tests for _build_dynamic_label method."""

    @pytest.fixture
    def mock_analyzer(self, monkeypatch: MonkeyPatch) -> Any:
        """Create analyzer with mocked Anthropic client."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from src.fpl_influencer_hivemind.analyzer.orchestrator import (
            FPLIntelligenceAnalyzer,
        )

        return FPLIntelligenceAnalyzer(verbose=False, save_prompts=False)

    def test_zero_transfers_label(self, mock_analyzer: Any) -> None:
        """Should generate roll label for 0 transfers."""
        label = mock_analyzer._build_dynamic_label(0, 0, 0)
        assert label == "Option A: 0 transfers (roll)"

    def test_one_transfer_no_hit(self, mock_analyzer: Any) -> None:
        """Should generate label for 1 transfer without hit."""
        label = mock_analyzer._build_dynamic_label(0, 1, 0)
        assert label == "Option A: 1 transfer (no hit)"

    def test_multiple_transfers_no_hit(self, mock_analyzer: Any) -> None:
        """Should pluralize for multiple transfers."""
        label = mock_analyzer._build_dynamic_label(1, 2, 0)
        assert label == "Option B: 2 transfers (no hit)"

    def test_transfers_with_hit(self, mock_analyzer: Any) -> None:
        """Should show hit cost in label."""
        label = mock_analyzer._build_dynamic_label(0, 2, 4)
        assert label == "Option A: 2 transfers (-4 hit)"

    def test_option_letter_sequence(self, mock_analyzer: Any) -> None:
        """Should use correct letter for index."""
        assert mock_analyzer._build_dynamic_label(0, 1, 0).startswith("Option A:")
        assert mock_analyzer._build_dynamic_label(1, 1, 0).startswith("Option B:")
        assert mock_analyzer._build_dynamic_label(2, 1, 0).startswith("Option C:")


class TestOptionRequest:
    """Tests for OptionRequest dataclass."""

    def test_create_option_request(self) -> None:
        """Should create OptionRequest with required fields."""
        req = OptionRequest(transfer_count=2, take_hit=True)
        assert req.transfer_count == 2
        assert req.take_hit is True

    def test_equality(self) -> None:
        """Should support equality comparison."""
        req1 = OptionRequest(transfer_count=1, take_hit=False)
        req2 = OptionRequest(transfer_count=1, take_hit=False)
        req3 = OptionRequest(transfer_count=1, take_hit=True)
        assert req1 == req2
        assert req1 != req3


class TestScoredModels:
    """Tests for ScoredPlayerRef and ScoredGapAnalysis models."""

    def test_scored_player_ref_defaults(self) -> None:
        """Should create ScoredPlayerRef with default severity."""
        from src.fpl_influencer_hivemind.types import ScoredPlayerRef

        player = ScoredPlayerRef(name="Haaland", position="FWD")
        assert player.severity == 0.0
        assert player.severity_factors == []

    def test_scored_player_ref_with_severity(self) -> None:
        """Should create ScoredPlayerRef with severity data."""
        from src.fpl_influencer_hivemind.types import ScoredPlayerRef

        player = ScoredPlayerRef(
            name="Isak",
            position="FWD",
            team="Newcastle",
            severity=8.5,
            severity_factors=["5/6 influencers", "high form"],
        )
        assert player.severity == 8.5
        assert len(player.severity_factors) == 2

    def test_scored_gap_analysis_defaults(self) -> None:
        """Should create ScoredGapAnalysis with default severities."""
        from src.fpl_influencer_hivemind.types import ScoredGapAnalysis

        gap = ScoredGapAnalysis(
            players_to_sell=[],
            players_missing=[],
            risk_flags=[],
            formation_gaps=[],
        )
        assert gap.captain_severity == 0.0
        assert gap.total_severity == 0.0

    def test_scored_gap_analysis_with_players(self) -> None:
        """Should create ScoredGapAnalysis with scored players."""
        from src.fpl_influencer_hivemind.types import (
            RiskFlag,
            ScoredGapAnalysis,
            ScoredPlayerRef,
        )

        gap = ScoredGapAnalysis(
            players_to_sell=[
                ScoredPlayerRef(
                    name="Mateta",
                    position="FWD",
                    severity=5.0,
                    severity_factors=["3/6 selling"],
                )
            ],
            players_missing=[
                ScoredPlayerRef(
                    name="Isak",
                    position="FWD",
                    severity=9.0,
                    severity_factors=["5/6 influencers"],
                )
            ],
            risk_flags=[RiskFlag(player="Saka", risk="Minor knock")],
            formation_gaps=["Low FWD depth"],
            captain_gap="Haaland",
            captain_severity=8.0,
            total_severity=22.0,
        )
        assert len(gap.players_missing) == 1
        assert gap.players_missing[0].severity == 9.0
        assert gap.total_severity == 22.0


class TestBuildStrategyLabel:
    """Tests for _build_strategy_label method."""

    @pytest.fixture
    def mock_analyzer(self, monkeypatch: MonkeyPatch) -> Any:
        """Create analyzer with mocked Anthropic client."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        from src.fpl_influencer_hivemind.analyzer.orchestrator import (
            FPLIntelligenceAnalyzer,
        )

        return FPLIntelligenceAnalyzer(verbose=False, save_prompts=False)

    def test_roll_label(self, mock_analyzer: Any) -> None:
        """Should generate roll label for 0 transfers."""
        from src.fpl_influencer_hivemind.types import TransferPlan

        transfers = TransferPlan(
            transfers=[],
            total_cost=0.0,
            new_itb=1.0,
            fts_used=0,
            fts_remaining=1,
            hit_cost=0,
            reasoning="Roll",
        )
        label = mock_analyzer._build_strategy_label("Conservative", transfers, [])
        assert label == "Conservative: Roll transfer (0 transfers, no hit)"

    def test_single_transfer_label(self, mock_analyzer: Any) -> None:
        """Should generate label with player name for single transfer."""
        from src.fpl_influencer_hivemind.types import Transfer, TransferPlan

        transfers = TransferPlan(
            transfers=[
                Transfer(
                    out_player="Mateta (FWD)",
                    out_team="Palace",
                    in_player="Isak (FWD)",
                    in_team="Newcastle",
                    in_price=9.0,
                    selling_price=6.5,
                    cost_delta=2.5,
                    backers=["FPL Mate"],
                )
            ],
            total_cost=2.5,
            new_itb=0.5,
            fts_used=1,
            fts_remaining=0,
            hit_cost=0,
            reasoning="Bring in Isak",
        )
        label = mock_analyzer._build_strategy_label("Balanced", transfers, ["Isak"])
        assert "Isak" in label
        assert "1 transfer" in label
        assert "no hit" in label

    def test_multiple_transfers_with_hit_label(self, mock_analyzer: Any) -> None:
        """Should show hit cost and multiple players."""
        from src.fpl_influencer_hivemind.types import Transfer, TransferPlan

        transfers = TransferPlan(
            transfers=[
                Transfer(
                    out_player="A (FWD)",
                    out_team="X",
                    in_player="B (FWD)",
                    in_team="Y",
                    in_price=8.0,
                    selling_price=7.0,
                    cost_delta=1.0,
                    backers=[],
                ),
                Transfer(
                    out_player="C (MID)",
                    out_team="X",
                    in_player="D (MID)",
                    in_team="Z",
                    in_price=9.0,
                    selling_price=8.0,
                    cost_delta=1.0,
                    backers=[],
                ),
            ],
            total_cost=2.0,
            new_itb=0.0,
            fts_used=2,
            fts_remaining=0,
            hit_cost=4,
            reasoning="2 transfers",
        )
        label = mock_analyzer._build_strategy_label(
            "Aggressive", transfers, ["B", "D", "E"]
        )
        assert "Aggressive" in label
        assert "2 transfers" in label
        assert "-4 hit" in label
        assert "+1 more" in label  # Because 3 gaps but only showing 2


class TestSeverityBar:
    """Tests for _severity_bar helper function."""

    def test_full_bar(self) -> None:
        """Should show full bar for max severity."""
        from src.fpl_influencer_hivemind.analyzer.report import _severity_bar

        bar = _severity_bar(10.0)
        assert bar == "██████████"

    def test_half_bar(self) -> None:
        """Should show half bar for 5.0 severity."""
        from src.fpl_influencer_hivemind.analyzer.report import _severity_bar

        bar = _severity_bar(5.0)
        assert bar == "█████░░░░░"

    def test_empty_bar(self) -> None:
        """Should show empty bar for 0.0 severity."""
        from src.fpl_influencer_hivemind.analyzer.report import _severity_bar

        bar = _severity_bar(0.0)
        assert bar == "░░░░░░░░░░"


class TestFormatGapSectionWithSeverity:
    """Tests for format_gap_section with severity data."""

    def test_format_scored_gap_with_severity(self) -> None:
        """Should display severity scores in gap section."""
        from src.fpl_influencer_hivemind.analyzer.report import format_gap_section
        from src.fpl_influencer_hivemind.types import (
            ScoredGapAnalysis,
            ScoredPlayerRef,
        )

        gap = ScoredGapAnalysis(
            players_to_sell=[],
            players_missing=[
                ScoredPlayerRef(
                    name="Isak",
                    position="FWD",
                    team="Newcastle",
                    severity=9.0,
                    severity_factors=["5/6 influencers", "high form"],
                )
            ],
            risk_flags=[],
            formation_gaps=[],
            captain_gap="Haaland",
            captain_severity=8.0,
            total_severity=17.0,
        )

        output = format_gap_section(gap)

        assert "Severity: 9/10" in output
        assert "████" in output  # Severity bar
        assert "5/6 influencers" in output
        assert "Total Severity:** 17.0" in output  # Markdown bold format

    def test_format_regular_gap_without_severity(self) -> None:
        """Should work with regular GapAnalysis without severity data."""
        from src.fpl_influencer_hivemind.analyzer.report import format_gap_section
        from src.fpl_influencer_hivemind.types import GapAnalysis, PlayerRef

        gap = GapAnalysis(
            players_to_sell=[],
            players_missing=[
                PlayerRef(name="Isak", position="FWD", team="Newcastle")
            ],
            risk_flags=[],
            formation_gaps=[],
            captain_gap="Haaland",
        )

        output = format_gap_section(gap)

        # Should not crash and should contain basic info
        assert "Isak" in output
        assert "Haaland" in output
        # Should NOT have severity bar for regular GapAnalysis
        assert "Severity:" not in output or "captain" in output.lower()
