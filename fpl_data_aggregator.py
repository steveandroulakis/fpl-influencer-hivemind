#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""CLI shim that invokes the Python aggregation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from fpl_influencer_hivemind.cli import default_transcript_prompt
from fpl_influencer_hivemind.pipeline import (
    AggregationError,
    aggregate,
    generate_unique_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect FPL and influencer data into a single JSON artifact.",
    )
    parser.add_argument("--team-id", type=int, required=True, help="FPL team ID to analyze")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional custom output path (unique suffix applied if path exists)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory for generated files (default: var/hivemind)",
    )
    parser.add_argument(
        "--auto-approve-transcripts",
        action="store_true",
        help="Skip transcript approval prompt",
    )
    parser.add_argument(
        "--skip-transcripts",
        action="store_true",
        help="Collect metadata only, without transcript downloads",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose subprocess output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        outcome = aggregate(
            team_id=args.team_id,
            artifacts_dir=args.artifacts_dir,
            auto_approve_transcripts=args.auto_approve_transcripts,
            fetch_transcripts=not args.skip_transcripts,
            verbose=args.verbose,
            prompt=None if args.auto_approve_transcripts else default_transcript_prompt,
        )
    except AggregationError as exc:
        print(f"Aggregation failed: {exc}")
        return 1

    output_path = outcome.result_path
    if args.output:
        destination = generate_unique_path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output_path.read_text(encoding="utf-8"), encoding="utf-8")
        output_path = destination

    print(f"Aggregation successful: {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())
