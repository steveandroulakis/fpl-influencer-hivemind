"""Command-line interface for orchestrating the FPL influencer pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from .pipeline import (
    PROJECT_ROOT,
    AggregationError,
    AggregationOutcome,
    LogCallback,
    LogLevel,
    aggregate,
    generate_unique_path,
)

if TYPE_CHECKING:
    from .types import ChannelDiscovery


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fpl-influencer-hivemind",
        description="Collect influencer data and optional AI analysis for FPL decisions",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Run the FPL data aggregation pipeline",
    )
    collect_parser.add_argument(
        "--team-id", type=int, required=True, help="FPL team ID"
    )
    collect_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory for storing aggregation outputs (default: var/hivemind)",
    )
    collect_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit path for aggregated JSON (unique suffix applied if needed)",
    )
    collect_parser.add_argument(
        "--auto-approve-transcripts",
        action="store_true",
        help="Skip confirmation before fetching transcripts",
    )
    collect_parser.add_argument(
        "--skip-transcripts",
        action="store_true",
        help="Collect metadata only without downloading transcripts",
    )
    collect_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display verbose logging from subprocesses",
    )

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Collect data then optionally run the intelligence analyzer",
    )
    pipeline_parser.add_argument(
        "--team-id", type=int, required=True, help="FPL team ID"
    )
    pipeline_parser.add_argument(
        "--free-transfers",
        type=int,
        default=1,
        choices=range(0, 6),
        help="Number of free transfers available when running analysis (0-5)",
    )
    pipeline_parser.add_argument(
        "--commentary",
        help="Optional high-priority user directive passed to the analyzer",
    )
    pipeline_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Directory for storing aggregation and analysis outputs",
    )
    pipeline_parser.add_argument(
        "--auto-approve-transcripts",
        action="store_true",
        help="Skip confirmation before fetching transcripts",
    )
    pipeline_parser.add_argument(
        "--auto-run-analysis",
        action="store_true",
        help="Run the analyzer without asking for confirmation",
    )
    pipeline_parser.add_argument(
        "--skip-transcripts",
        action="store_true",
        help="Collect metadata only without downloading transcripts",
    )
    pipeline_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Display verbose logging from subprocesses",
    )

    return parser


def _make_console_logger(verbose: bool) -> LogCallback:
    """Create a logging callback that writes pipeline events to the console."""

    prefixes: dict[LogLevel, str] = {
        "success": "✅ ",
        "warning": "⚠️ ",
        "error": "❌ ",
        "debug": "[debug] ",
    }

    def _log(message: str, level: LogLevel) -> None:
        if level == "debug" and not verbose:
            return
        prefix = prefixes.get(level, "")
        stream = sys.stderr if level == "error" else sys.stdout
        print(f"{prefix}{message}", file=stream)

    return _log


def _summarize_discoveries(discoveries: Sequence[ChannelDiscovery]) -> str:
    lines: list[str] = []
    for item in discoveries:
        channel_name = item.channel.get("name", "Unknown channel")
        if item.result:
            title = item.result.get("title", "Unknown title")
            confidence = item.result.get("confidence", "?")
            lines.append(
                f"✅ {channel_name}: {title} (confidence {confidence})"
            )
        else:
            reason = item.error or "No match"
            lines.append(f"❌ {channel_name}: {reason}")
    return "\n".join(lines)


def default_transcript_prompt(discoveries: Sequence[ChannelDiscovery]) -> bool:
    """Prompt the user to confirm transcript fetching."""

    summary = _summarize_discoveries(discoveries)
    if summary:
        print("\n=== Selected Videos ===")
        print(summary)
        print("")

    successful_count = sum(1 for item in discoveries if item.result)
    print(f"\n📥 Ready to fetch {successful_count} transcripts")

    while True:
        response = input("Proceed with transcript fetching? [y/N]: ").strip().lower()
        if response in {"y", "yes"}:
            print("✓ Starting transcript fetch...\n")
            return True
        if response in {"", "n", "no"}:
            print("✗ Skipping transcript fetch\n")
            return False
        print("Please enter 'y' or 'n'.")


def _run_collect(args: argparse.Namespace) -> int:
    log = _make_console_logger(verbose=args.verbose)
    try:
        outcome = aggregate(
            team_id=args.team_id,
            artifacts_dir=args.artifacts_dir,
            auto_approve_transcripts=args.auto_approve_transcripts,
            fetch_transcripts=not args.skip_transcripts,
            verbose=args.verbose,
            prompt=None if args.auto_approve_transcripts else default_transcript_prompt,
            log=log,
        )
    except AggregationError as exc:
        print(f"Aggregation failed: {exc}", file=sys.stderr)
        return 1

    result_path = outcome.result_path
    if args.output:
        destination = args.output
        if destination.suffix != ".json":
            destination = destination.with_suffix(".json")
        destination = generate_unique_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            result_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        result_path = destination

    _print_collect_summary(outcome, result_path)
    return 0


def _print_collect_summary(outcome: AggregationOutcome, path: Path) -> None:
    payload = {
        "team_id": outcome.team_id,
        "gameweek": outcome.gameweek_id,
        "channels_processed": outcome.channels_processed,
        "videos_discovered": outcome.videos_discovered,
        "transcripts_retrieved": outcome.transcripts_retrieved,
        "result_path": str(path),
    }
    print(json.dumps(payload, indent=2))


def _confirm(prompt: str) -> bool:
    """Prompt for confirmation, flushing any buffered stdin first."""

    with suppress(ImportError, OSError):
        import termios  # type: ignore

        termios.tcflush(sys.stdin, termios.TCIFLUSH)

    response = input(prompt).strip().lower()
    return response in {"y", "yes"}


def _run_analyzer(
    *,
    input_path: Path,
    output_path: Path,
    free_transfers: int,
    verbose: bool,
    commentary: str | None = None,
) -> int:
    args = [
        sys.executable,
        str(PROJECT_ROOT / "fpl_intelligence_analyzer.py"),
        "--input",
        str(input_path),
        "--output-file",
        str(output_path),
        "--free-transfers",
        str(free_transfers),
    ]
    if commentary and commentary.strip():
        args.extend(["--commentary", commentary])
    if verbose:
        args.append("--verbose")
    try:
        subprocess.run(args, check=True, text=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        print(f"Analysis failed: {exc.stderr}", file=sys.stderr)
        return 1
    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    log = _make_console_logger(verbose=args.verbose)
    try:
        outcome = aggregate(
            team_id=args.team_id,
            artifacts_dir=args.artifacts_dir,
            auto_approve_transcripts=args.auto_approve_transcripts,
            fetch_transcripts=not args.skip_transcripts,
            verbose=args.verbose,
            prompt=None if args.auto_approve_transcripts else default_transcript_prompt,
            log=log,
        )
    except AggregationError as exc:
        print(f"Aggregation failed: {exc}", file=sys.stderr)
        return 1

    _print_collect_summary(outcome, outcome.result_path)

    if args.auto_run_analysis:
        run_analysis = True
    else:
        run_analysis = _confirm(
            "Run intelligence analyzer now? This may consume Anthropic quota. [y/N]: "
        )

    if not run_analysis:
        print(
            "Analysis skipped. Use the 'collect' command output with fpl_intelligence_analyzer.py later."
        )
        return 0

    artifacts_dir = args.artifacts_dir or outcome.result_path.parent
    report_name = f"analysis_gw{outcome.gameweek_id:02d}_team{outcome.team_id}_{outcome.result_path.stem}.md"
    report_path = generate_unique_path(Path(artifacts_dir) / report_name)

    status = _run_analyzer(
        input_path=outcome.result_path,
        output_path=report_path,
        free_transfers=args.free_transfers,
        verbose=args.verbose,
        commentary=args.commentary,
    )
    if status == 0:
        print(f"Analysis complete: {report_path}")
    return status


def main(argv: Sequence[str] | None = None) -> int:
    if argv is not None and not isinstance(argv, Sequence):
        argv = list(argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "collect":
        return _run_collect(args)
    if args.command == "pipeline":
        return _run_pipeline(args)
    parser.error(f"Unknown command {args.command}")
    return 1


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
