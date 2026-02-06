"""FPL Intelligence Analyzer module.

This module provides deterministic FPL analysis with LLM extraction only.
"""

from src.fpl_influencer_hivemind.analyzer.cli import main
from src.fpl_influencer_hivemind.analyzer.models import ChannelAnalysis
from src.fpl_influencer_hivemind.analyzer.simple_orchestrator import SimpleFPLAnalyzer

__all__ = ["ChannelAnalysis", "SimpleFPLAnalyzer", "main"]
