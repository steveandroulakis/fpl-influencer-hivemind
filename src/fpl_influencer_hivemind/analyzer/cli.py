"""CLI for the FPL Intelligence Analyzer."""

import argparse
import sys

from src.fpl_influencer_hivemind.analyzer.simple_orchestrator import (
    SimpleFPLAnalyzer,
)


def main() -> int:
    """Main entry point for the FPL intelligence analyzer."""
    parser = argparse.ArgumentParser(
        description=(
            "FPL Intelligence Analyzer - Deterministic recommendations with "
            "LLM extraction only"
        ),
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
    parser.add_argument(
        "--narrative",
        action="store_true",
        help="Generate a narrative-only summary from the computed report",
    )

    args = parser.parse_args()

    try:
        analyzer = SimpleFPLAnalyzer(
            verbose=args.verbose, save_prompts=not args.no_save_prompts
        )
        analyzer.run_analysis(
            args.input,
            args.output_file,
            args.free_transfers,
            commentary=args.commentary,
            narrative=args.narrative,
        )
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


__all__ = ["main"]
