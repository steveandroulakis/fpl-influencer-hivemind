"""Core pipeline orchestration for the FPL influencer hivemind."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NotRequired, Protocol, TypedDict, cast

from .youtube import VideoPickerError, select_single_channel


class CommandRunner(Protocol):
    """Protocol describing a callable that executes an external command."""

    def run(
        self, args: Sequence[str], *, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Execute *args* in *cwd* and return the completed process."""


class SubprocessRunner:
    """`CommandRunner` implementation that shells out via :mod:`subprocess`."""

    def run(
        self, args: Sequence[str], *, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=None if cwd is None else str(cwd),
            check=True,
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,  # Prevent subprocesses from consuming stdin
        )


class AggregationError(RuntimeError):
    """Raised when a pipeline step fails."""


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FPL_DIR = PROJECT_ROOT / "fpl"
YOUTUBE_TITLES_DIR = PROJECT_ROOT / "youtube-titles"
YOUTUBE_TRANSCRIPTS_DIR = PROJECT_ROOT / "youtube-transcript"
DEFAULT_CHANNELS_PATH = YOUTUBE_TITLES_DIR / "channels.json"
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


class VideoResult(TypedDict, total=False):
    """Partial schema emitted by ``fpl_video_picker.py``."""

    channel_name: str
    video_id: str
    title: str
    url: str
    confidence: float
    published_at: str
    published_at_formatted: NotRequired[str]
    reasoning: NotRequired[str]
    matched_signals: NotRequired[list[str]]
    gameweek: NotRequired[int]
    generated_at: NotRequired[str]


class TranscriptEntry(TypedDict):
    """Structure stored in the aggregated JSON for each transcript."""

    video_id: str
    transcript: str


class GameweekInfo(TypedDict, total=False):
    """Subset of the gameweek payload returned by ``get_current_gameweek``."""

    id: int
    name: str
    is_current: bool
    is_next: bool
    is_past: bool
    finished: bool
    data_checked: bool
    calculation_method: str
    now_utc: str
    now_local: str
    deadline_time_utc: str | None
    deadline_time_local: str | None
    time_until_deadline: str | None
    time_since_deadline: str | None
    deadline_passed: bool | None


class MyTeamPayload(TypedDict, total=False):
    """Minimal schema for the ``get_my_team`` JSON output."""

    summary: dict[str, object]
    current_picks: list[dict[str, object]]
    team_value: dict[str, object]


class ChannelsFile(TypedDict):
    """Structure of ``youtube-titles/channels.json``."""

    channels: list[ChannelConfig]


class ChannelConfig(TypedDict, total=False):
    """Channel metadata used for discovery."""

    name: str
    url: str
    description: NotRequired[str]


PromptCallback = Callable[[Sequence["ChannelDiscovery"]], bool]


@dataclass(slots=True)
class ChannelDiscovery:
    """Outcome of attempting to identify a video for a channel."""

    channel: ChannelConfig
    result: VideoResult | None
    error: str | None = None


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


def _ensure_scripts_present() -> None:
    expected = [
        FPL_DIR / "get_current_gameweek.py",
        FPL_DIR / "get_top_ownership.py",
        FPL_DIR / "get_my_team.py",
        YOUTUBE_TITLES_DIR / "fpl_video_picker.py",
        YOUTUBE_TRANSCRIPTS_DIR / "fpl_transcript.py",
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise AggregationError(f"Missing required scripts: {joined}")


def _build_script_command(script_path: Path, args: Sequence[str]) -> list[str]:
    if UV_BIN:
        return [UV_BIN, "run", str(script_path), *args]
    return [sys.executable, str(script_path), *args]


def _run_python_script(
    runner: CommandRunner,
    script_path: Path,
    args: Sequence[str],
) -> None:
    try:
        runner.run(_build_script_command(script_path, args), cwd=script_path.parent)
    except subprocess.CalledProcessError as exc:
        raise AggregationError(
            f"Command failed: {' '.join(exc.cmd)}\n{exc.stderr}"
        ) from exc


def _read_json(path: Path) -> dict[str, object]:
    try:
        return cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise AggregationError(f"Expected JSON output missing: {path}") from exc


def _read_json_list(path: Path) -> list[dict[str, object]]:
    try:
        return cast(
            "list[dict[str, object]]", json.loads(path.read_text(encoding="utf-8"))
        )
    except FileNotFoundError as exc:
        raise AggregationError(f"Expected JSON output missing: {path}") from exc


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


def _load_channels(path: Path) -> list[ChannelConfig]:
    if not path.exists():
        raise AggregationError(f"Channels configuration missing: {path}")
    payload = cast("ChannelsFile", json.loads(path.read_text(encoding="utf-8")))
    return payload.get("channels", [])


def _collect_fpl_data(
    runner: CommandRunner,
    temp_dir: Path,
    team_id: int,
) -> tuple[GameweekInfo, list[dict[str, object]], MyTeamPayload]:
    gameweek_path = temp_dir / "gameweek.json"
    ownership_path = temp_dir / "top_players.json"
    my_team_path = temp_dir / "my_team.json"

    print("  ↳ Fetching current gameweek info...", flush=True)
    _run_python_script(
        runner,
        FPL_DIR / "get_current_gameweek.py",
        ["--out", str(gameweek_path)],
    )

    print("  ↳ Fetching top 150 players by ownership...", flush=True)
    _run_python_script(
        runner,
        FPL_DIR / "get_top_ownership.py",
        ["--limit", "150", "--format", "json", "--out", str(ownership_path)],
    )

    print(f"  ↳ Fetching your team data (entry {team_id})...", flush=True)
    _run_python_script(
        runner,
        FPL_DIR / "get_my_team.py",
        [
            "--entry-id",
            str(team_id),
            "--format",
            "json",
            "--out",
            str(my_team_path),
        ],
    )

    gameweek = cast("GameweekInfo", _read_json(gameweek_path))
    top_players = _read_json_list(ownership_path)
    my_team = cast("MyTeamPayload", _read_json(my_team_path))
    print("  ✅ FPL data collected\n", flush=True)
    return gameweek, top_players, my_team


def _extract_video_id(url: str) -> str | None:
    match = re.search(r"v=([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    if "youtu.be/" in url:
        candidate = url.rsplit("/", 1)[-1]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
            return candidate
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    return None


def _discover_videos(
    channels: Iterable[ChannelConfig],
    *,
    gameweek_id: int,
    days: int,
    max_per_channel: int,
    verbose: bool,
) -> list[ChannelDiscovery]:
    discoveries: list[ChannelDiscovery] = []
    video_logger = logging.getLogger("fpl_influencer_hivemind.video_picker")
    if not video_logger.handlers:
        video_logger.addHandler(logging.NullHandler())
    video_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    for channel in channels:
        channel_name = channel.get("name", "Unknown channel")
        channel_url = channel.get("url")
        if not channel_url:
            error = "Channel configuration missing 'url'"
            print(f"❌ {channel_name}: {error}")
            discoveries.append(
                ChannelDiscovery(channel=channel, result=None, error=error)
            )
            continue

        print(f"🔍 Discovering video for {channel_name}...", flush=True)

        try:
            selection = select_single_channel(
                channel_name=channel_name,
                channel_url=channel_url,
                gameweek=gameweek_id,
                days_back=days,
                max_per_channel=max_per_channel,
                logger=video_logger,
            )
        except VideoPickerError as exc:
            error = str(exc)
            print(f"❌ {channel_name}: {error}", flush=True)
            if verbose:
                import traceback
                print(f"   Traceback: {traceback.format_exc()}", flush=True)
            discoveries.append(
                ChannelDiscovery(channel=channel, result=None, error=error)
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive
            error = f"Video discovery failed: {exc}"
            print(f"❌ {channel_name}: {error}", flush=True)
            if verbose:
                import traceback
                print(f"   Traceback: {traceback.format_exc()}", flush=True)
            discoveries.append(
                ChannelDiscovery(channel=channel, result=None, error=error)
            )
            continue

        video = selection.picked
        if video is None:
            error = f"No video selected for {channel_name}"
            print(f"❌ {channel_name}: {error}", flush=True)
            discoveries.append(
                ChannelDiscovery(channel=channel, result=None, error=error)
            )
            continue

        video_id = _extract_video_id(video.url) or ""
        payload: VideoResult = {
            "channel_name": channel_name,
            "video_id": video_id,
            "title": video.title,
            "url": video.url,
            "confidence": selection.confidence,
            "published_at": video.published_at.astimezone(UTC).isoformat(),
            "published_at_formatted": video.published_at.astimezone(UTC).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
            "reasoning": selection.reasoning,
            "matched_signals": selection.matched_signals,
            "gameweek": gameweek_id,
            "generated_at": datetime.now(UTC).isoformat(),
        }

        print(f"✅ {channel_name}: {video.title}", flush=True)

        # Show alternatives if verbose
        if verbose and selection.alternatives:
            print(f"   📋 Alternatives considered:", flush=True)
            for alt in selection.alternatives[:3]:
                print(f"      • {alt.title[:80]}", flush=True)

        discoveries.append(
            ChannelDiscovery(channel=channel, result=payload, error=None)
        )

    return discoveries


def _summarize_discoveries(discoveries: Sequence[ChannelDiscovery]) -> str:
    if not isinstance(discoveries, Sequence):
        discoveries = list(discoveries)
    lines: list[str] = []
    for item in discoveries:
        channel_name = item.channel.get("name", "Unknown channel")
        if item.result:
            result = item.result
            title = result.get("title", "Unknown title")
            confidence = result.get("confidence", "?")
            lines.append(f"✅ {channel_name}: {title} (confidence {confidence})")
        else:
            reason = item.error or "No match"
            lines.append(f"❌ {channel_name}: {reason}")
    return "\n".join(lines)


def _print_discoveries(discoveries: Sequence[ChannelDiscovery]) -> None:
    summary = _summarize_discoveries(discoveries)
    if summary:
        print("\n=== Selected Videos ===")
        print(summary)
        print("")


def default_transcript_prompt(discoveries: Sequence[ChannelDiscovery]) -> bool:
    """Prompt the user to confirm transcript fetching."""

    _print_discoveries(discoveries)

    successful_count = sum(1 for d in discoveries if d.result)
    print(f"\n📥 Ready to fetch {successful_count} transcripts", flush=True)

    while True:
        response = input("Proceed with transcript fetching? [y/N]: ").strip().lower()
        if response in {"y", "yes"}:
            print("✓ Starting transcript fetch...\n", flush=True)
            return True
        if response in {"", "n", "no"}:
            print("✗ Skipping transcript fetch\n", flush=True)
            return False
        print("Please enter 'y' or 'n'.")


def _normalize_transcript(content: str) -> str:
    return " ".join(content.split())


def _fetch_transcripts(
    runner: CommandRunner,
    temp_dir: Path,
    discoveries: Iterable[ChannelDiscovery],
    *,
    delay_seconds: float,
    verbose: bool,
) -> dict[str, TranscriptEntry]:
    transcripts: dict[str, TranscriptEntry] = {}
    delay_applied = False
    discoveries_list = list(discoveries)
    for item in discoveries_list:
        if not item.result:
            continue
        video_id = item.result["video_id"]
        channel_name = item.result.get(
            "channel_name", item.channel.get("name", "Channel")
        )
        output_path = temp_dir / f"transcript_{video_id}.txt"
        args = [
            f"--id={video_id}",
            "--format",
            "txt",
            "--out",
            str(output_path),
            "--timeout",
            "120",  # Increased from 60s - allows for retries and slow downloads
            "--delay",
            "5",
            "--random-delay",
        ]
        if verbose:
            args.append("--verbose")

        if delay_applied and delay_seconds > 0:
            time.sleep(delay_seconds)
        delay_applied = True

        channel_label = channel_name or item.channel.get("name", "Channel")
        print(f"Fetching transcript for {channel_label} ({video_id})...", flush=True)

        try:
            _run_python_script(
                runner,
                YOUTUBE_TRANSCRIPTS_DIR / "fpl_transcript.py",
                args,
            )
        except AggregationError as exc:
            error_msg = str(exc).split("\n")[0]  # Show first line of error
            print(f"❌ Failed to fetch transcript for {channel_label}: {error_msg}", flush=True)
            transcripts[channel_name] = {
                "video_id": video_id,
                "transcript": f"Transcript fetch failed: {error_msg}",
            }
            continue

        if not output_path.exists():
            print(f"⚠️  Transcript file missing for {channel_label}; skipping.", flush=True)
            transcripts[channel_name] = {
                "video_id": video_id,
                "transcript": "Transcript not available",
            }
            continue

        raw_text = output_path.read_text(encoding="utf-8")
        print(f"✅ Transcript fetched for {channel_label}.", flush=True)
        transcripts[channel_name] = {
            "video_id": video_id,
            "transcript": _normalize_transcript(raw_text),
        }

    return transcripts


def aggregate(
    *,
    team_id: int,
    runner: CommandRunner | None = None,
    channels: Sequence[ChannelConfig] | None = None,
    auto_approve_transcripts: bool = False,
    fetch_transcripts: bool = True,
    artifacts_dir: Path | None = None,
    discovery_days: int = 7,
    max_per_channel: int = 6,
    transcript_delay: float = 10.0,
    verbose: bool = False,
    prompt: PromptCallback | None = None,
) -> AggregationOutcome:
    """Run the aggregation pipeline and return metadata about the results."""

    print("🚀 Starting FPL Influencer Hivemind pipeline...", flush=True)

    _ensure_env_loaded()
    _ensure_scripts_present()

    if team_id <= 0:
        raise AggregationError("team_id must be a positive integer")

    runner = runner or SubprocessRunner()
    artifacts_dir = _ensure_artifacts_dir(artifacts_dir or DEFAULT_ARTIFACTS_DIR)
    channels_list = list(channels or _load_channels(DEFAULT_CHANNELS_PATH))
    if not channels_list:
        raise AggregationError("Channel configuration did not yield any channels")

    print(f"📊 Fetching FPL data for team {team_id}...", flush=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    with TemporaryDirectory(prefix="hivemind_", dir=str(artifacts_dir)) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        gameweek, top_players, my_team = _collect_fpl_data(runner, temp_dir, team_id)
        source_gameweek_id = int(gameweek.get("id", 0))
        requested_gameweek_id = (
            source_gameweek_id + 1
            if gameweek.get("is_current", False) and source_gameweek_id
            else source_gameweek_id
        )

        transcripts: dict[str, TranscriptEntry] = {}

        if requested_gameweek_id == source_gameweek_id:
            active_gameweek_id = source_gameweek_id
            print(f"🎬 Discovering videos for GW{active_gameweek_id} from {len(channels_list)} channels...\n", flush=True)
            discoveries = _discover_videos(
                channels_list,
                gameweek_id=active_gameweek_id,
                days=discovery_days,
                max_per_channel=max_per_channel,
                verbose=verbose,
            )
            successful_discoveries = [item for item in discoveries if item.result]
            fallback_used = False
        else:
            print(f"🎬 Discovering videos for GW{requested_gameweek_id} from {len(channels_list)} channels...\n", flush=True)
            discoveries = _discover_videos(
                channels_list,
                gameweek_id=requested_gameweek_id,
                days=discovery_days,
                max_per_channel=max_per_channel,
                verbose=verbose,
            )
            successful_discoveries = [item for item in discoveries if item.result]
            if successful_discoveries:
                active_gameweek_id = requested_gameweek_id
                fallback_used = False
            else:
                print(
                    f"⚠️  No matching videos found for requested gameweek "
                    f"{requested_gameweek_id}; falling back to current week {source_gameweek_id}.\n", flush=True
                )
                active_gameweek_id = source_gameweek_id
                print(f"🎬 Discovering videos for GW{active_gameweek_id} from {len(channels_list)} channels...\n", flush=True)
                discoveries = _discover_videos(
                    channels_list,
                    gameweek_id=active_gameweek_id,
                    days=discovery_days,
                    max_per_channel=max_per_channel,
                    verbose=verbose,
                )
                successful_discoveries = [item for item in discoveries if item.result]
                fallback_used = True

        _print_discoveries(discoveries)

        if fetch_transcripts and successful_discoveries:
            if auto_approve_transcripts:
                proceed = True
            else:
                prompt_cb = prompt or default_transcript_prompt
                proceed = prompt_cb(discoveries)

            if proceed:
                print(f"\n📥 Starting transcript fetch for {len(successful_discoveries)} videos...", flush=True)
                transcripts = _fetch_transcripts(
                    runner,
                    temp_dir,
                    successful_discoveries,
                    delay_seconds=transcript_delay,
                    verbose=verbose,
                )
                print(f"✅ Transcript fetch complete. Retrieved {len(transcripts)} transcripts.", flush=True)

        gameweek_entry = {
            "current": active_gameweek_id,
            "source": source_gameweek_id,
            "requested": requested_gameweek_id,
            "fallback_used": fallback_used,
        }
        video_results: list[VideoResult] = [
            cast("VideoResult", item.result)
            for item in discoveries
            if item.result is not None
        ]

        aggregate_payload: dict[str, object] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "team_id": team_id,
            "gameweek": gameweek_entry,
            "fpl_data": {
                "gameweek_info": gameweek,
                "top_players": top_players,
                "my_team": my_team,
            },
            "youtube_analysis": {
                "channels_processed": len(channels_list),
                "videos_discovered": len(successful_discoveries),
                "transcripts_retrieved": len(transcripts),
                "video_results": video_results,
                "transcripts": transcripts,
            },
            "summary": {
                "total_channels": len(channels_list),
                "failed_discoveries": len(
                    [item for item in discoveries if item.result is None]
                ),
                "failed_transcripts": len(successful_discoveries) - len(transcripts),
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
        result_path.write_text(
            json.dumps(aggregate_payload, indent=2), encoding="utf-8"
        )

    return AggregationOutcome(
        team_id=team_id,
        gameweek_id=active_gameweek_id,
        result_path=result_path,
        video_results=video_results,
        transcripts=transcripts,
        channels_processed=len(channels_list),
        videos_discovered=len(successful_discoveries),
        transcripts_retrieved=len(transcripts),
    )


__all__ = sorted(
    [
        "AggregationError",
        "AggregationOutcome",
        "CommandRunner",
        "PROJECT_ROOT",
        "SubprocessRunner",
        "aggregate",
        "default_transcript_prompt",
        "generate_unique_path",
    ]
)
UV_BIN = os.environ.get("UV_BIN") or shutil.which("uv")
