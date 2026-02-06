#!/usr/bin/env python3
"""
FPL Intelligence Analyzer

Processes FPL data aggregator output through deterministic analysis to generate
transfer and captain recommendations. LLM usage is limited to transcript
extraction and optional narrative summarization.

Usage:
    ./fpl_intelligence_analyzer.py --input fpl_analysis_results_clean.json
    ./fpl_intelligence_analyzer.py --input data.json --output-file analysis.md --verbose
    ./fpl_intelligence_analyzer.py --input data.json --output-file analysis.md --free-transfers 2
    ./fpl_intelligence_analyzer.py --input data.json -o analysis.md -ft 0  # Must take hit or roll
    ./fpl_intelligence_analyzer.py --input data.json --commentary "Plan to wildcard, recommend only wildcard path"
    ./fpl_intelligence_analyzer.py --input data.json --output-file analysis.md --narrative

Use `--commentary` to inject a high-priority user directive that the analysis must follow.
"""

import sys

from src.fpl_influencer_hivemind.analyzer.cli import main

if __name__ == "__main__":
    sys.exit(main())
