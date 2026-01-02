"""Core pipeline orchestration for the FPL influencer hivemind."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Literal, cast

from .services import fpl as fpl_service
from .services import transcripts as transcript_service
from .services.discovery import DiscoveryStrategy, HeuristicDiscoveryStrategy
from .services.fpl import FPLServiceError
from .services.transcripts import TranscriptServiceError
from .types import (
    ChannelConfig,
    ChannelDiscovery,
    ChannelsFile,
    GameweekInfo,
    MyTeamPayload,
    TranscriptEntry,
    TranscriptErrorEntry,
    VideoResult,
)


class AggregationError(RuntimeError):
    """Raised when a pipeline step fails."""


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHANNELS_RESOURCE = "fpl_influencer_hivemind.data"
DEFAULT_CHANNELS_FILENAME = "channels.json"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "var" / "hivemind"
_ENV_LOADED = False


@dataclass(slots=True)
class AggregationOutcome:
    """Metadata about a completed aggregation run."""

    team_id: int
    gameweek_id: int
    result_path: Path
    video_results: list[VideoResult]
    transcripts: dict[str, TranscriptEntry]
    channels_processed: int
    videos_discovered: int
    transcripts_retrieved: int
    transcript_errors: dict[str, TranscriptErrorEntry] = field(default_factory=dict)


LogLevel = Literal["info", "success", "warning", "error", "debug"]
LogCallback = Callable[[str, LogLevel], None]


PromptCallback = Callable[[Sequence[ChannelDiscovery]], bool]


def _emit(log: LogCallback | None, message: str, level: LogLevel = "info") -> None:
    """Send a log message to the provided callback if available."""

    if log is not None:
        log(message, level)


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from an ``.env`` file into ``os.environ``."""

    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        elif line.startswith("export\t"):
            line = line[len("export\t") :].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if value and value[0] in {'"', "'"} and value[-1] == value[0]:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()

        value = os.path.expandvars(value)

        os.environ[key] = value


def _ensure_env_loaded() -> None:
    """Load the project ``.env`` file once per interpreter session."""

    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _load_env_file(PROJECT_ROOT / ".env")
    _ENV_LOADED = True


def generate_unique_path(base: Path) -> Path:
    """Return a unique path derived from ``base`` without overwriting existing files."""

    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.stem}-{suffix}{base.suffix}")
        suffix += 1
    return candidate


def _ensure_artifacts_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_channel_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "", name.replace(" ", "_"))
    return sanitized or "channel"


def _load_channels(path: Path | None = None) -> list[ChannelConfig]:
    if path is None:
        with (
            resources.files(DEFAULT_CHANNELS_RESOURCE)
            .joinpath(DEFAULT_CHANNELS_FILENAME)
            .open("r", encoding="utf-8") as handle
        ):
            payload = cast("ChannelsFile", json.load(handle))
    else:
        if not path.exists():
            raise AggregationError(f"Channels configuration missing: {path}")
        payload = cast("ChannelsFile", json.loads(path.read_text(encoding="utf-8")))
    return payload.get("channels", [])


def _collect_fpl_data(
    team_id: int, log: LogCallback | None
) -> tuple[
    GameweekInfo,
    list[dict[str, object]],
    MyTeamPayload,
    dict[str, list[dict[str, object]]],
]:
    _emit(log, "Fetching current gameweek info...", "info")
    try:
        gameweek = cast("GameweekInfo", fpl_service.get_current_gameweek_info())
    except FPLServiceError as exc:
        raise AggregationError(str(exc)) from exc

    _emit(log, "Fetching top 150 players by ownership...", "info")
    try:
        top_players = fpl_service.get_top_ownership(limit=150)
    except FPLServiceError as exc:
        raise AggregationError(str(exc)) from exc

    _emit(log, f"Fetching your team data (entry {team_id})...", "info")
    try:
        my_team = cast("MyTeamPayload", fpl_service.get_my_team(team_id))
    except FPLServiceError as exc:
        raise AggregationError(str(exc)) from exc

    _emit(log, "Fetching transfer momentum data...", "info")
    try:
        transfer_momentum = fpl_service.get_transfer_momentum(limit=10)
    except FPLServiceError as exc:
        raise AggregationError(str(exc)) from exc

    _emit(log, "FPL data collected", "success")
    return gameweek, top_players, my_team, transfer_momentum


def _discover_videos(
    channels: Iterable[ChannelConfig],
    *,
    gameweek_id: int,
    days: int,
    max_per_channel: int,
    verbose: bool,
    log: LogCallback | None,
    strategy: DiscoveryStrategy,
) -> list[ChannelDiscovery]:
    discoveries: list[ChannelDiscovery] = []

    for channel in channels:
        channel_name = channel.get("name", "Unknown channel")
        _emit(log, f"Discovering video for {channel_name}...", "info")
        discovery = strategy.discover(
            channel=channel,
            gameweek_id=gameweek_id,
            days=days,
            max_per_channel=max_per_channel,
            verbose=verbose,
        )

        if discovery.result:
            title = discovery.result.get("title", "Unknown title")
            _emit(log, f"Video selected for {channel_name}: {title}", "success")
            if verbose and discovery.alternatives:
                _emit(log, "Alternatives considered:", "debug")
                for alt in discovery.alternatives:
                    _emit(log, f"Alternative: {alt}", "debug")
        else:
            error = discovery.error or "No match"
            _emit(log, f"{channel_name}: {error}", "error")

        discoveries.append(discovery)

    return discoveries


def _fetch_transcripts(
    discoveries: Iterable[ChannelDiscovery],
    *,
    delay_seconds: float,
    verbose: bool,
    log: LogCallback | None,
) -> tuple[dict[str, TranscriptEntry], dict[str, TranscriptErrorEntry]]:
    transcripts: dict[str, TranscriptEntry] = {}
    transcript_errors: dict[str, TranscriptErrorEntry] = {}
    delay_applied = False
    discoveries_list = list(discoveries)
    for item in discoveries_list:
        if not item.result:
            continue
        video_id = item.result["video_id"]
        channel_name = item.result.get(
            "channel_name", item.channel.get("name", "Channel")
        )
        if delay_applied and delay_seconds > 0:
            time.sleep(delay_seconds)
        delay_applied = True

        channel_label = channel_name or item.channel.get("name", "Channel")
        _emit(
            log,
            f"Fetching transcript for {channel_label} ({video_id})...",
            "info",
        )

        try:
            transcript_payload = transcript_service.fetch_transcript(
                video_id,
                timeout=300.0,
                delay=5.0,
                random_delay=True,
                verbose=verbose,
            )
        except TranscriptServiceError as exc:
            error_msg = str(exc)
            _emit(
                log,
                f"Failed to fetch transcript for {channel_label}: {error_msg}",
                "error",
            )
            transcript_errors[video_id] = {
                "video_id": video_id,
                "channel_name": channel_label,
                "error": error_msg,
            }
            continue
        _emit(
            log,
            f"Transcript fetched for {channel_label}.",
            "success",
        )
        transcripts[video_id] = transcript_payload

    return transcripts, transcript_errors


def aggregate(
    *,
    team_id: int,
    channels: Sequence[ChannelConfig] | None = None,
    auto_approve_transcripts: bool = False,
    fetch_transcripts: bool = True,
    artifacts_dir: Path | None = None,
    discovery_days: int = 7,
    max_per_channel: int = 6,
    transcript_delay: float = 10.0,
    verbose: bool = False,
    prompt: PromptCallback | None = None,
    log: LogCallback | None = None,
    discovery_strategy: DiscoveryStrategy | None = None,
) -> AggregationOutcome:
    """Run the aggregation pipeline and return metadata about the results."""

    _emit(log, "Starting FPL Influencer Hivemind pipeline...", "info")

    _ensure_env_loaded()

    if team_id <= 0:
        raise AggregationError("team_id must be a positive integer")

    artifacts_dir = _ensure_artifacts_dir(artifacts_dir or DEFAULT_ARTIFACTS_DIR)
    channels_list = list(channels or _load_channels())
    if not channels_list:
        raise AggregationError("Channel configuration did not yield any channels")

    _emit(log, f"Fetching FPL data for team {team_id}...", "info")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    gameweek, top_players, my_team, transfer_momentum = _collect_fpl_data(team_id, log)
    source_gameweek_id = int(gameweek.get("id", 0))
    requested_gameweek_id = (
        source_gameweek_id + 1
        if gameweek.get("is_current", False) and source_gameweek_id
        else source_gameweek_id
    )

    transcripts: dict[str, TranscriptEntry] = {}
    transcript_errors: dict[str, TranscriptErrorEntry] = {}

    strategy = discovery_strategy or HeuristicDiscoveryStrategy()

    if requested_gameweek_id == source_gameweek_id:
        active_gameweek_id = source_gameweek_id
        _emit(
            log,
            f"Discovering videos for GW{active_gameweek_id} from {len(channels_list)} channels...",
            "info",
        )
        discoveries = _discover_videos(
            channels_list,
            gameweek_id=active_gameweek_id,
            days=discovery_days,
            max_per_channel=max_per_channel,
            verbose=verbose,
            log=log,
            strategy=strategy,
        )
        successful_discoveries = [item for item in discoveries if item.result]
        fallback_used = False
    else:
        _emit(
            log,
            f"Discovering videos for GW{requested_gameweek_id} from {len(channels_list)} channels...",
            "info",
        )
        discoveries = _discover_videos(
            channels_list,
            gameweek_id=requested_gameweek_id,
            days=discovery_days,
            max_per_channel=max_per_channel,
            verbose=verbose,
            log=log,
            strategy=strategy,
        )
        successful_discoveries = [item for item in discoveries if item.result]
        if successful_discoveries:
            active_gameweek_id = requested_gameweek_id
            fallback_used = False
        else:
            _emit(
                log,
                (
                    "No matching videos found for requested gameweek "
                    f"{requested_gameweek_id}; falling back to current week {source_gameweek_id}."
                ),
                "warning",
            )
            active_gameweek_id = source_gameweek_id
            _emit(
                log,
                f"Discovering videos for GW{active_gameweek_id} from {len(channels_list)} channels...",
                "info",
            )
            discoveries = _discover_videos(
                channels_list,
                gameweek_id=active_gameweek_id,
                days=discovery_days,
                max_per_channel=max_per_channel,
                verbose=verbose,
                log=log,
                strategy=strategy,
            )
            successful_discoveries = [item for item in discoveries if item.result]
            fallback_used = True

    _emit(
        log,
        (
            "Video discovery complete: "
            f"{len(successful_discoveries)} successful out of {len(channels_list)} channels."
        ),
        "info",
    )

    if not successful_discoveries:
        _emit(
            log,
            "No videos were discovered for the configured channels.",
            "warning",
        )

    if fetch_transcripts and successful_discoveries:
        if auto_approve_transcripts:
            proceed = True
        elif prompt is not None:
            proceed = prompt(discoveries)
        else:
            _emit(
                log,
                "Transcript fetch skipped: prompt callback not provided.",
                "warning",
            )
            proceed = False

        if proceed:
            _emit(
                log,
                f"Starting transcript fetch for {len(successful_discoveries)} videos...",
                "info",
            )
            transcripts, transcript_errors = _fetch_transcripts(
                successful_discoveries,
                delay_seconds=transcript_delay,
                verbose=verbose,
                log=log,
            )
            _emit(
                log,
                f"Transcript fetch complete. Retrieved {len(transcripts)} transcripts.",
                "success",
            )
            if transcript_errors:
                _emit(
                    log,
                    f"Transcript fetch errors: {len(transcript_errors)} failures.",
                    "warning",
                )
        else:
            _emit(log, "Transcript fetch skipped by user.", "info")
    elif fetch_transcripts:
        _emit(
            log,
            "Transcript fetch skipped: no successful video discoveries.",
            "info",
        )
    else:
        _emit(
            log,
            "Transcript fetching disabled for this run.",
            "info",
        )

    gameweek_entry = {
        "current": active_gameweek_id,
        "source": source_gameweek_id,
        "requested": requested_gameweek_id,
        "fallback_used": fallback_used,
    }
    video_results: list[VideoResult] = [
        item.result for item in discoveries if item.result is not None
    ]

    aggregate_payload: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "team_id": team_id,
        "gameweek": gameweek_entry,
        "fpl_data": {
            "gameweek_info": gameweek,
            "top_players": top_players,
            "my_team": my_team,
            "transfer_momentum": transfer_momentum,
        },
        "youtube_analysis": {
            "channels_processed": len(channels_list),
            "videos_discovered": len(successful_discoveries),
            "transcripts_retrieved": len(transcripts),
            "video_results": video_results,
            "transcripts": transcripts,
            "transcript_errors": transcript_errors,
        },
        "summary": {
            "total_channels": len(channels_list),
            "failed_discoveries": len(
                [item for item in discoveries if item.result is None]
            ),
            "failed_transcripts": len(transcript_errors),
            "success_rate": (
                f"{len(successful_discoveries) * 100 // len(channels_list)}%"
                if channels_list
                else "0%"
            ),
        },
    }

    default_filename = (
        f"gw{active_gameweek_id:02d}_team{team_id}_{timestamp}_aggregation.json"
    )
    result_path = generate_unique_path(artifacts_dir / default_filename)
    result_path.write_text(json.dumps(aggregate_payload, indent=2), encoding="utf-8")

    return AggregationOutcome(
        team_id=team_id,
        gameweek_id=active_gameweek_id,
        result_path=result_path,
        video_results=video_results,
        transcripts=transcripts,
        transcript_errors=transcript_errors,
        channels_processed=len(channels_list),
        videos_discovered=len(successful_discoveries),
        transcripts_retrieved=len(transcripts),
    )


__all__ = [
    "PROJECT_ROOT",
    "AggregationError",
    "AggregationOutcome",
    "aggregate",
    "generate_unique_path",
]
