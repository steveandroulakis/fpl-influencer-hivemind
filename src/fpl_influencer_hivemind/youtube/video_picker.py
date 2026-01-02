"""YouTube video discovery utilities used by the hivemind pipeline."""

import argparse
import json
import logging
import os
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import anthropic
from googleapiclient.discovery import build  # type: ignore[import-untyped]
from googleapiclient.errors import HttpError  # type: ignore[import-untyped]
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHANNELS_CONFIG = PROJECT_ROOT / "youtube-titles" / "channels.json"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
DEFAULT_TEMPERATURE = 0.0


class VideoPickerError(RuntimeError):
    """Raised when video discovery cannot produce a result."""


@dataclass
class VideoItem:
    """Represents a YouTube video with metadata."""

    title: str
    url: str
    published_at: datetime
    channel_name: str
    description: str = ""
    normalized_title: str = field(init=False)

    def __post_init__(self) -> None:
        self.normalized_title = self.normalize_text(self.title)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for matching - lowercase, strip whitespace."""
        return text.lower().strip()


class AnthropicChannelResponse(BaseModel):
    """Pydantic model for single channel response."""

    channel_name: str
    chosen_index: int
    chosen_url: str
    confidence: float
    matched_signals: list[str]
    reasoning: str


class AnthropicResponse(BaseModel):
    """Pydantic model for Anthropic API response validation."""

    channels: list[AnthropicChannelResponse]


@dataclass
class ChannelResult:
    """Result of video selection for a single channel."""

    channel_name: str
    channel_url: str
    picked: VideoItem | None
    alternatives: list[VideoItem]
    confidence: float = 0.0
    reasoning: str = ""
    matched_signals: list[str] = field(default_factory=list)
    heuristic_score: float = 0.0


@dataclass
class SelectionResult:
    """Final result of video selection process for all channels."""

    channel_results: list[ChannelResult]
    gameweek: int | None
    generated_at: datetime
    model: str
    heuristic_notes: dict[str, Any]


def select_single_channel(
    *,
    channel_name: str,
    channel_url: str,
    gameweek: int | None,
    days_back: int,
    max_per_channel: int,
    anthropic_model: str = DEFAULT_ANTHROPIC_MODEL,  # noqa: ARG001
    temperature: float = DEFAULT_TEMPERATURE,  # noqa: ARG001
    logger: logging.Logger | None = None,
) -> ChannelResult:
    """Discover the best team-selection video for a single channel."""

    logger = logger or logging.getLogger(__name__)
    print("  ↳ Fetching videos from YouTube API...", flush=True)
    collector = FPLVideoCollector(
        max_per_channel=max_per_channel, days_back=days_back, logger=logger
    )
    videos_by_channel = collector.collect_videos_by_channel([channel_url])
    videos = videos_by_channel.get(channel_url, [])
    print(f"  ↳ Found {len(videos)} recent videos", flush=True)

    if not videos:
        raise VideoPickerError(f"No recent videos found for {channel_name}")

    heuristic_filter = HeuristicFilter(gameweek=gameweek, logger=logger)
    candidates, _notes = heuristic_filter.filter_and_rank(
        videos, max_candidates=max_per_channel
    )
    if not candidates:
        raise VideoPickerError(
            f"No videos passed heuristic filtering for {channel_name}"
        )

    selected_video = candidates[0]

    # Use heuristic score as confidence (0.0-1.0 range)
    heuristic_score_raw = heuristic_filter.calculate_score(selected_video)
    confidence = min(0.95, heuristic_score_raw / 10.0)  # Scale to 0-1, cap at 0.95

    # Generate reasoning based on matched keywords
    matched_keywords = []
    text = f"{selected_video.normalized_title} {selected_video.description.lower()}"
    for keyword in heuristic_filter.POSITIVE_KEYWORDS:
        if keyword in text:
            matched_keywords.append(keyword)

    reasoning = "Heuristic selection based on keyword matching"
    if gameweek:
        detected_gw = heuristic_filter.extract_gameweek_number(text)
        if detected_gw == gameweek:
            reasoning = f"Explicit GW{gameweek} team selection video with clear title indicating team selection and transfers"
            matched_keywords.append(f"gw{gameweek}")

    matched_signals: list[str] = matched_keywords[:3]  # Top 3 signals

    # Anthropic ranking disabled for performance - heuristics work well
    # try:
    #     ranker = AnthropicRanker(
    #         model=anthropic_model, temperature=temperature, logger=logger
    #     )
    #     ranked = ranker.rank_videos_by_channel({channel_url: candidates}, gameweek)
    #     if ranked and channel_url in ranked:
    #         selected_video, anthropic_response = ranked[channel_url]
    #         confidence = anthropic_response.confidence
    #         reasoning = anthropic_response.reasoning
    #         matched_signals = anthropic_response.matched_signals
    # except Exception as exc:  # pragma: no cover - fallback safety
    #     logger.debug("Anthropic ranking unavailable, using heuristic result: %s", exc)

    heuristic_score = heuristic_filter.calculate_score(selected_video)
    alternatives = [video for video in candidates if video is not selected_video][:3]

    return ChannelResult(
        channel_name=channel_name,
        channel_url=channel_url,
        picked=selected_video,
        alternatives=alternatives,
        confidence=confidence,
        reasoning=reasoning,
        matched_signals=matched_signals,
        heuristic_score=heuristic_score,
    )


class FPLVideoCollector:  # pragma: no cover - hits live YouTube Data API
    """Handles YouTube video collection with resilient fetching."""

    DEFAULT_CHANNELS: ClassVar[list[str]] = [
        "https://www.youtube.com/@FPLRaptor",
        "https://www.youtube.com/channel/UCweDAlFm2LnVcOqaFU4_AGA",  # FPL Mate
        "https://www.youtube.com/channel/UCxeOc7eFxq37yW_Nc-69deA",  # FPL Andy
        "https://www.youtube.com/fplfocal",
        "https://www.youtube.com/channel/UCcPWnCj5AKC19HaySZjb25g",  # FP Harry
    ]

    def __init__(
        self,
        max_per_channel: int = 6,
        days_back: int = 14,
        logger: logging.Logger | None = None,
    ):
        self.max_per_channel = max_per_channel
        self.days_back = days_back
        self.logger = logger or logging.getLogger(__name__)
        self.cutoff_date = datetime.now(UTC) - timedelta(days=days_back)

    def _resolve_channel_id(self, yt_service: Any, channel_url: str) -> str | None:
        """Resolve various YouTube channel URL formats to channel ID."""
        try:
            if "@" in channel_url:
                # Handle URL: https://www.youtube.com/@FPLRaptor
                handle = channel_url.split("@")[1]
                print(f"      • Looking up @{handle}...", flush=True)
                response = (
                    yt_service.channels()
                    .list(part="id", forHandle=f"@{handle}")
                    .execute()
                )

                if response.get("items"):
                    return response["items"][0]["id"]  # type: ignore[no-any-return]

            elif "channel/" in channel_url:
                # Channel ID URL: https://www.youtube.com/channel/UCweDAlFm2LnVcOqaFU4_AGA
                print("      • Using direct channel ID", flush=True)
                return channel_url.split("channel/")[1]

            else:
                # Custom URL: https://www.youtube.com/fplfocal
                # Extract the identifier after the last slash
                search_query = channel_url.split("/")[-1]

                # Try search API to find the channel
                print(f"      • Searching for '{search_query}'...", flush=True)
                search_response = (
                    yt_service.search()
                    .list(part="snippet", type="channel", q=search_query, maxResults=5)
                    .execute()
                )

                if search_response.get("items"):
                    # Return the first result's channel ID
                    return search_response["items"][0]["snippet"]["channelId"]  # type: ignore[no-any-return]

        except HttpError as e:
            self.logger.error(f"API error resolving channel ID for {channel_url}: {e}")
        except Exception as e:
            self.logger.error(f"Error resolving channel ID for {channel_url}: {e}")

        return None

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError, HttpError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def _fetch_channel_videos(self, channel_url: str) -> list[VideoItem]:
        """Fetch videos from a single channel using YouTube Data API v3."""
        import time as time_module

        start_time = time_module.time()
        videos = []
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            self.logger.error("YOUTUBE_API_KEY environment variable not found")
            return []

        try:
            print("    → Initializing YouTube API client...", flush=True)
            import socket

            socket.setdefaulttimeout(30)  # 30 second timeout for API calls
            yt = build("youtube", "v3", developerKey=api_key)

            # Resolve channel URL to channel ID
            print("    → Resolving channel ID from URL...", flush=True)
            channel_id = self._resolve_channel_id(yt, channel_url)
            if not channel_id:
                self.logger.error(f"Could not resolve channel ID for: {channel_url}")
                return []

            # Get channel info and uploads playlist ID
            print("    → Fetching channel metadata...", flush=True)
            channel_response = (
                yt.channels()
                .list(part="snippet,contentDetails", id=channel_id)
                .execute()
            )

            if not channel_response.get("items"):
                self.logger.error(f"Channel not found: {channel_id}")
                return []

            channel_info = channel_response["items"][0]
            channel_name = channel_info["snippet"]["title"]
            uploads_playlist_id = channel_info["contentDetails"]["relatedPlaylists"][
                "uploads"
            ]

            self.logger.debug(f"Fetching videos from {channel_name} (ID: {channel_id})")
            print(
                f"    → Fetching recent uploads (need {self.max_per_channel})...",
                flush=True,
            )

            # Fetch videos from uploads playlist
            # IMPORTANT: Only fetch a small batch and stop early to avoid scanning entire channel history
            all_videos: list[VideoItem] = []
            next_page_token = None
            total_fetched = 0
            filtered_count = 0
            max_attempts = 2  # Only fetch 2 pages max (up to 100 videos) to avoid scanning thousands

            attempt = 0
            while len(all_videos) < self.max_per_channel and attempt < max_attempts:
                attempt += 1
                # Only request what we need plus a small buffer
                remaining_needed = self.max_per_channel - len(all_videos)
                batch_size = min(
                    50, remaining_needed + 10
                )  # Small buffer for filtering
                print(
                    f"      • API call {attempt}/{max_attempts}: requesting {batch_size} videos (have {len(all_videos)}/{self.max_per_channel})...",
                    flush=True,
                )
                videos_response = (
                    yt.playlistItems()
                    .list(
                        part="snippet,contentDetails",
                        playlistId=uploads_playlist_id,
                        maxResults=batch_size,
                        pageToken=next_page_token,
                    )
                    .execute()
                )

                items = videos_response.get("items", [])
                total_fetched += len(items)
                if not items:
                    break

                consecutive_old = 0
                for item in items:
                    try:
                        # Parse published date
                        published_str = item["snippet"]["publishedAt"]
                        # YouTube API returns RFC 3339 format: 2024-08-23T15:30:00Z
                        published_at = datetime.fromisoformat(
                            published_str.replace("Z", "+00:00")
                        )

                        # Skip videos that are too old
                        if published_at < self.cutoff_date:
                            filtered_count += 1
                            consecutive_old += 1
                            # If we hit 20 consecutive old videos, stop - we've gone too far back
                            if consecutive_old >= 20:
                                print(
                                    "      • Stopping: found 20 consecutive old videos",
                                    flush=True,
                                )
                                break
                            continue

                        consecutive_old = 0  # Reset counter when we find a recent video

                        video_id = item["contentDetails"]["videoId"]
                        video_url = f"https://www.youtube.com/watch?v={video_id}"

                        video_item = VideoItem(
                            title=item["snippet"]["title"],
                            url=video_url,
                            published_at=published_at,
                            channel_name=channel_name,
                            description=item["snippet"]["description"] or "",
                        )

                        all_videos.append(video_item)
                        self.logger.debug(f"  Added: {video_item.title[:60]}...")

                        # Stop if we have enough videos
                        if len(all_videos) >= self.max_per_channel:
                            break

                    except (ValueError, KeyError) as e:
                        self.logger.debug(
                            f"  Skipped video {item.get('snippet', {}).get('title', 'Unknown')}: {e}"
                        )
                        continue

                # Break if we hit too many old videos
                if consecutive_old >= 20:
                    break

                next_page_token = videos_response.get("nextPageToken")
                if not next_page_token:
                    break

            videos = all_videos[: self.max_per_channel]

            elapsed = time_module.time() - start_time
            if total_fetched > 0:
                print(
                    f"    → Found {total_fetched} videos, filtered {filtered_count} (too old), kept {len(videos)} in {elapsed:.1f}s",
                    flush=True,
                )

        except HttpError as e:
            error_msg = f"YouTube API error for {channel_url}: {e}"
            self.logger.error(error_msg)
            print(f"    ✗ {error_msg}", flush=True)
            return []
        except Exception as e:
            error_msg = f"Failed to fetch from {channel_url}: {e}"
            self.logger.error(error_msg)
            print(f"    ✗ {error_msg}", flush=True)
            return []

        self.logger.info(f"Collected {len(videos)} videos from {channel_name}")
        return videos

    def collect_videos_by_channel(
        self, channel_urls: list[str]
    ) -> dict[str, list[VideoItem]]:
        """Collect videos from all provided channels, grouped by channel (in parallel)."""
        videos_by_channel = {}
        total_videos = 0

        # Process channels in parallel for better performance
        with ThreadPoolExecutor(max_workers=min(5, len(channel_urls))) as executor:
            future_to_url = {
                executor.submit(self._fetch_channel_videos, url): url
                for url in channel_urls
            }

            for future in as_completed(future_to_url):
                channel_url = future_to_url[future]
                try:
                    videos = future.result()
                    if videos:
                        # Sort each channel's videos by publication date (newest first)
                        videos.sort(key=lambda v: v.published_at, reverse=True)
                        videos_by_channel[channel_url] = videos
                        total_videos += len(videos)
                    else:
                        videos_by_channel[channel_url] = []
                except Exception as exc:
                    self.logger.error(
                        f"Channel {channel_url} generated an exception: {exc}"
                    )
                    videos_by_channel[channel_url] = []

        self.logger.info(
            f"Total videos collected: {total_videos} across {len(videos_by_channel)} channels"
        )
        return videos_by_channel


class HeuristicFilter:  # pragma: no cover - exercised indirectly via select_single_channel
    """Pre-filters videos using keyword-based heuristics."""

    POSITIVE_KEYWORDS: ClassVar[list[str]] = [
        "team selection",
        "my team",
        "team reveal",
        "final team",
        "gw",
        "gameweek",
        "wildcard",
        "free hit",
        "bench boost",
        "triple captain",
        "draft",
        "picks",
        "starting xi",
    ]

    NEGATIVE_KEYWORDS: ClassVar[list[str]] = [
        "deadline stream",
        "watchalong",
        "price change",
        "press conference",
        "reaction",
        "news",
        "match reaction",
        "live stream",
        "q&a",
        "chat",
    ]

    def __init__(
        self, gameweek: int | None = None, logger: logging.Logger | None = None
    ):
        self.gameweek = gameweek
        self.logger = logger or logging.getLogger(__name__)

    def extract_gameweek_number(self, text: str) -> int | None:
        """Extract gameweek number from text."""
        gw_patterns = [r"\bgw\s*(\d+)\b", r"\bgameweek\s*(\d+)\b", r"\bgw(\d+)\b"]

        for pattern in gw_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        return None

    def calculate_score(self, video: VideoItem) -> float:
        """Calculate heuristic score for a video."""
        score = 0.0
        text = f"{video.normalized_title} {video.description.lower()}"

        # Positive keywords boost
        for keyword in self.POSITIVE_KEYWORDS:
            if keyword in text:
                if keyword in ["team selection", "team reveal", "final team"]:
                    score += 3.0  # Strong indicators
                elif keyword in ["gw", "gameweek"]:
                    score += 2.0
                else:
                    score += 1.0

        # Negative keywords reduce score
        for keyword in self.NEGATIVE_KEYWORDS:
            if keyword in text:
                score -= 2.0

        # Gameweek-specific bonus
        if self.gameweek:
            detected_gw = self.extract_gameweek_number(text)
            if detected_gw == self.gameweek:
                score += 5.0  # Strong bonus for matching gameweek
            elif detected_gw and abs(detected_gw - self.gameweek) <= 1:
                score += 1.0  # Small bonus for adjacent gameweeks

        # Recency bonus (newer videos get slight preference)
        days_old = (datetime.now(UTC) - video.published_at).days
        if days_old <= 1:
            score += 1.0
        elif days_old <= 3:
            score += 0.5

        return max(0.0, score)  # Don't allow negative scores

    def filter_and_rank(
        self, videos: list[VideoItem], max_candidates: int = 12
    ) -> tuple[list[VideoItem], dict[str, Any]]:
        """Filter videos and return top candidates with scoring details."""
        scored_videos = []

        for video in videos:
            score = self.calculate_score(video)
            if score > 0:  # Only keep videos with positive scores
                scored_videos.append((video, score))

        # Sort by score descending, then by recency
        scored_videos.sort(key=lambda x: (x[1], x[0].published_at), reverse=True)

        # Take top candidates
        top_candidates = [video for video, _ in scored_videos[:max_candidates]]

        heuristic_notes = {
            "rules": {
                "positive_keywords": self.POSITIVE_KEYWORDS,
                "negative_keywords": self.NEGATIVE_KEYWORDS,
                "gameweek_filter": self.gameweek,
            },
            "scores": {
                video.url: score for video, score in scored_videos[:max_candidates]
            },
        }

        self.logger.info(
            f"Heuristic filter: {len(top_candidates)} candidates from {len(videos)} videos"
        )
        return top_candidates, heuristic_notes


class AnthropicRanker:  # pragma: no cover - requires live Anthropic API interaction
    """Uses Anthropic Claude to intelligently rank video candidates."""

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        logger: logging.Logger | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.logger = logger or logging.getLogger(__name__)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")

        self.client = anthropic.Anthropic(api_key=api_key)

    def _create_batch_prompt(
        self, channels_candidates: dict[str, list[VideoItem]], gameweek: int | None
    ) -> str:
        """Create the prompt for Claude to rank videos for all channels."""
        channels_json = {}

        for _channel_url, candidates in channels_candidates.items():
            if not candidates:
                continue

            channel_name = candidates[0].channel_name
            candidates_json = []

            for i, video in enumerate(candidates):
                candidates_json.append(
                    {
                        "index": i,
                        "title": video.title,
                        "url": video.url,
                        "published_at": video.published_at.isoformat(),
                        "desc": video.description[:200] + "..."
                        if len(video.description) > 200
                        else video.description,
                    }
                )

            channels_json[channel_name] = candidates_json

        gameweek_context = f" for gameweek {gameweek}" if gameweek else ""

        return f"""You are an FPL content classifier. Return ONLY valid JSON with no extra commentary.

For each YouTube channel, select the single best "team selection" video{gameweek_context}.

CHANNEL CANDIDATES:
{json.dumps(channels_json, indent=2)}

Typical team-selection signals:
- "team selection", "team reveal", "final team", "GW{gameweek or 'X'} team"
- "wildcard draft", "free hit team", "bench boost team"

Selection rules:
- Prefer explicit gameweek matches when available.
- Prefer the most recent relevant video when multiple match.
- If no clear match exists, pick the closest and set a low confidence.

Return this exact JSON schema with one result per channel:
```json
{{
  "channels": [
    {{
      "channel_name": "FPL Mate",
      "chosen_index": 0,
      "chosen_url": "https://youtube.com/watch?v=...",
      "confidence": 0.85,
      "matched_signals": ["team selection", "gw5"],
      "reasoning": "Clear team selection video with gameweek number"
    }}
  ]
}}
```

Analyze each channel separately and choose the best team-selection video for each."""

    def rank_videos_by_channel(
        self,
        channels_candidates: dict[str, list[VideoItem]],
        gameweek: int | None = None,
    ) -> dict[str, tuple[VideoItem, AnthropicChannelResponse] | None]:
        """Use Claude to rank videos for each channel and return the best choices."""
        results: dict[str, tuple[VideoItem, AnthropicChannelResponse]] = {}

        # Filter out channels with no candidates
        valid_channels = {
            url: candidates
            for url, candidates in channels_candidates.items()
            if candidates
        }

        if not valid_channels:
            return {}

        try:
            prompt = self._create_batch_prompt(valid_channels, gameweek)

            total_candidates = sum(
                len(candidates) for candidates in valid_channels.values()
            )
            self.logger.debug(
                f"Calling Anthropic API with {total_candidates} candidates across {len(valid_channels)} channels"
            )

            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,  # Increased for multiple channels
                temperature=self.temperature,
                system="You are an FPL content classifier. You strictly return only valid JSON and no extra commentary.",
                messages=[{"role": "user", "content": prompt}],
            )

            # Type narrowing: Anthropic should return TextBlock for text content
            if not message.content or not hasattr(message.content[0], "text"):
                raise ValueError("Expected text response from Anthropic API")
            # Anthropic API returns TextBlock for text messages
            from anthropic.types import TextBlock

            if isinstance(message.content[0], TextBlock):
                response_text = message.content[0].text.strip()
            else:
                raise ValueError(f"Unexpected content type: {type(message.content[0])}")
            self.logger.debug(f"Raw API response: {response_text}")

            # Parse JSON response
            parsed_response = self._parse_batch_response(response_text, valid_channels)
            if not parsed_response:
                return {}

            # Map responses back to channels
            for channel_response in parsed_response.channels:
                # Find the channel URL by matching channel name
                matching_channel_url = None
                for url, candidates in valid_channels.items():
                    if (
                        candidates
                        and candidates[0].channel_name == channel_response.channel_name
                    ):
                        matching_channel_url = url
                        break

                if matching_channel_url and 0 <= channel_response.chosen_index < len(
                    valid_channels[matching_channel_url]
                ):
                    chosen_video = valid_channels[matching_channel_url][
                        channel_response.chosen_index
                    ]
                    results[matching_channel_url] = (chosen_video, channel_response)
                    self.logger.info(
                        f"Claude selected for {channel_response.channel_name}: {chosen_video.title[:60]}... (confidence: {channel_response.confidence})"
                    )
                else:
                    self.logger.warning(
                        f"Could not map response for channel: {channel_response.channel_name}"
                    )

            # Convert to expected return type (values can be None)
            return dict(results)

        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            # Fallback to heuristic selections
            return {}

    def _parse_batch_response(
        self, response_text: str, channels_candidates: dict[str, list[VideoItem]]
    ) -> AnthropicResponse | None:
        """Parse and validate Claude's JSON response for batch processing."""
        try:
            # Remove code fences if present
            if "```" in response_text:
                json_match = re.search(
                    r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
                )
                if json_match:
                    response_text = json_match.group(1)

            data = json.loads(response_text)
            response = AnthropicResponse(**data)

            # Validate each channel response
            for channel_response in response.channels:
                # Find matching channel
                matching_candidates = None
                for candidates in channels_candidates.values():
                    if (
                        candidates
                        and candidates[0].channel_name == channel_response.channel_name
                    ):
                        matching_candidates = candidates
                        break

                if not matching_candidates:
                    self.logger.error(
                        f"No candidates found for channel: {channel_response.channel_name}"
                    )
                    continue

                # Validate chosen_index and chosen_url
                if not (0 <= channel_response.chosen_index < len(matching_candidates)):
                    self.logger.error(
                        f"Invalid chosen_index {channel_response.chosen_index} for {channel_response.channel_name}"
                    )
                    continue

                expected_url = matching_candidates[channel_response.chosen_index].url
                if channel_response.chosen_url != expected_url:
                    self.logger.error(
                        f"URL mismatch for {channel_response.channel_name}: got {channel_response.chosen_url}, expected {expected_url}"
                    )
                    continue

            return response

        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            self.logger.error(f"Failed to parse Anthropic response: {e}")
            return None


def load_channels_from_file(
    file_path: str, logger: logging.Logger
) -> list[str]:  # pragma: no cover - CLI helper
    """Load channel URLs from a text file."""
    try:
        with Path(file_path).open() as f:
            channels = []
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    channels.append(line)
        logger.info(f"Loaded {len(channels)} channels from {file_path}")
        return channels
    except FileNotFoundError:
        logger.error(f"Channels file not found: {file_path}")
        return []


def load_channels_config(
    config_path: str, logger: logging.Logger
) -> dict[str, dict[str, str]]:  # pragma: no cover - CLI helper
    """Load channels from JSON config file."""
    try:
        with Path(config_path).open() as f:
            config = json.load(f)
        channels = {}
        for channel in config["channels"]:
            channels[channel["name"]] = {
                "url": channel["url"],
                "description": channel.get("description", ""),
            }
        logger.info(
            f"Loaded {len(channels)} channels from config: {', '.join(channels.keys())}"
        )
        return channels
    except FileNotFoundError:
        logger.error(f"Channels config file not found: {config_path}")
        return {}
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Invalid channels config format: {e}")
        return {}


def setup_logging(
    verbose: bool = False,
) -> logging.Logger:  # pragma: no cover - CLI helper
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def cli_main(
    argv: Sequence[str] | None = None,
) -> int:  # pragma: no cover - CLI wrapper
    """Main entry point for the FPL video picker script."""
    parser = argparse.ArgumentParser(
        description="Find the most likely FPL team selection video from YouTube channels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --gameweek 5
  %(prog)s --max-per-channel 10 --out results.json
  %(prog)s --channels-file my_channels.txt --verbose
  %(prog)s --single-channel "FPL Raptor" --gameweek 2 --out raptor.json

Returns one selected video per channel, analyzing each channel's videos individually.
Use --single-channel for parallel processing in shell scripts.
        """,
    )

    parser.add_argument(
        "--channels-file", help="Path to file containing channel URLs (one per line)"
    )
    parser.add_argument(
        "--single-channel",
        help="Process only a single channel by name (requires --channels-config)",
    )
    parser.add_argument(
        "--channels-config",
        default=str(DEFAULT_CHANNELS_CONFIG),
        help="Path to channels.json config file (default: project youtube-titles/channels.json)",
    )
    parser.add_argument(
        "--max-per-channel",
        type=int,
        default=6,
        help="Maximum videos to fetch per channel (default: 6)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Only consider videos from the last N days (default: 14)",
    )
    parser.add_argument(
        "--gameweek", "-gw", type=int, help="Target gameweek number for selection bias"
    )
    parser.add_argument(
        "--anthropic-model",
        default=DEFAULT_ANTHROPIC_MODEL,
        help=f"Anthropic model to use (default: {DEFAULT_ANTHROPIC_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for Anthropic API (default: 0.0)",
    )
    parser.add_argument("--out", help="Path to write JSON results file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    args = parser.parse_args(argv)

    # Setup logging
    logger = setup_logging(args.verbose)
    logger.info("Starting FPL Video Picker")

    try:
        # Load channels
        if args.single_channel:
            # Single channel mode
            channels_config = load_channels_config(args.channels_config, logger)
            if not channels_config:
                return 1

            if args.single_channel not in channels_config:
                logger.error(
                    f"Channel '{args.single_channel}' not found in config. Available: {', '.join(channels_config.keys())}"
                )
                return 1

            channels = [channels_config[args.single_channel]["url"]]
            logger.info(f"Processing single channel: {args.single_channel}")

        elif args.channels_file:
            channels = load_channels_from_file(args.channels_file, logger)
            if not channels:
                return 1
        else:
            channels = FPLVideoCollector.DEFAULT_CHANNELS
            logger.info(f"Using {len(channels)} default channels")

        # Collect videos by channel
        collector = FPLVideoCollector(
            max_per_channel=args.max_per_channel, days_back=args.days, logger=logger
        )
        videos_by_channel = collector.collect_videos_by_channel(channels)

        if not any(videos_by_channel.values()):
            logger.error("No videos found matching criteria")
            return 2

        # Process each channel individually
        heuristic_filter = HeuristicFilter(gameweek=args.gameweek, logger=logger)
        channel_results = []
        channels_candidates = {}
        heuristic_notes = {}

        # Apply heuristic filtering per channel
        for channel_url, videos in videos_by_channel.items():
            if not videos:
                channel_name = channel_url.split("/")[-1]
                channel_results.append(
                    ChannelResult(
                        channel_name=channel_name,
                        channel_url=channel_url,
                        picked=None,
                        alternatives=[],
                    )
                )
                continue

            candidates, channel_heuristic_notes = heuristic_filter.filter_and_rank(
                videos, max_candidates=6
            )
            channels_candidates[channel_url] = candidates
            heuristic_notes[videos[0].channel_name] = channel_heuristic_notes

            logger.info(
                f"{videos[0].channel_name}: {len(candidates)} candidates after filtering"
            )

        # Rank with Anthropic (batch processing)
        anthropic_results = {}
        try:
            ranker = AnthropicRanker(
                model=args.anthropic_model, temperature=args.temperature, logger=logger
            )

            anthropic_results = ranker.rank_videos_by_channel(
                channels_candidates, args.gameweek
            )

        except Exception as e:
            logger.warning(f"Anthropic ranking failed: {e}, using heuristic fallback")

        # Create channel results
        for channel_url, candidates in channels_candidates.items():
            if not candidates:
                continue

            channel_name = candidates[0].channel_name

            # Check if we have Anthropic results for this channel
            if channel_url in anthropic_results:
                anthropic_result = anthropic_results[channel_url]
                if anthropic_result is None:
                    continue
                selected_video, anthropic_response = anthropic_result
                confidence = anthropic_response.confidence
                reasoning = anthropic_response.reasoning
                matched_signals = anthropic_response.matched_signals
            else:
                # Fallback to heuristic choice (top candidate)
                selected_video = candidates[0]
                confidence = 0.0
                reasoning = "Heuristic fallback - top scored candidate"
                matched_signals = []

            # Calculate heuristic score for the selected video
            heuristic_score = heuristic_filter.calculate_score(selected_video)

            channel_results.append(
                ChannelResult(
                    channel_name=channel_name,
                    channel_url=channel_url,
                    picked=selected_video,
                    alternatives=candidates[1:4]
                    if selected_video == candidates[0]
                    else candidates[:3],  # Top 3 alternatives
                    confidence=confidence,
                    reasoning=reasoning,
                    matched_signals=matched_signals,
                    heuristic_score=heuristic_score,
                )
            )

        if not channel_results:
            logger.error("No results generated for any channel")
            return 2

        # Create result
        result = SelectionResult(
            channel_results=channel_results,
            gameweek=args.gameweek,
            generated_at=datetime.now(UTC),
            model=args.anthropic_model,
            heuristic_notes=heuristic_notes,
        )

        # Print human-readable summary
        if args.single_channel:
            # Single channel mode - simplified output
            channel_result = channel_results[0]
            if channel_result.picked:
                print("\n🎯 SELECTED VIDEO:")
                print(f"Channel: {channel_result.channel_name}")
                print(f"Title: {channel_result.picked.title}")
                print(f"URL: {channel_result.picked.url}")
                print(
                    f"Video ID: {channel_result.picked.url.split('v=')[1] if 'v=' in channel_result.picked.url else 'N/A'}"
                )
                print(f"Confidence: {channel_result.confidence:.2f}")
                print(
                    f"Published: {channel_result.picked.published_at.strftime('%Y-%m-%d %H:%M UTC')}"
                )
            else:
                print(f"\n❌ NO VIDEO FOUND for {channel_result.channel_name}")
        else:
            # Multi-channel mode - existing output format
            print("\n🎯 SELECTED VIDEOS BY CHANNEL:")
            print(f"{'=' * 80}")

            for i, channel_result in enumerate(channel_results, 1):
                if channel_result.picked:
                    print(f"\n{i}. {channel_result.channel_name}:")
                    print(f"   Title: {channel_result.picked.title}")
                    print(
                        f"   Published: {channel_result.picked.published_at.strftime('%Y-%m-%d %H:%M UTC')}"
                    )
                    print(f"   URL: {channel_result.picked.url}")
                    print(
                        f"   Confidence: {channel_result.confidence:.2f} | Heuristic Score: {channel_result.heuristic_score:.1f}"
                    )
                    print(f"   Reasoning: {channel_result.reasoning}")
                    if channel_result.matched_signals:
                        print(
                            f"   Signals: {', '.join(channel_result.matched_signals)}"
                        )
                else:
                    print(
                        f"\n{i}. {channel_result.channel_name}: No suitable videos found"
                    )

            print(f"\n{'=' * 80}")
            successful_selections = len([r for r in channel_results if r.picked])
            print(
                f"Successfully selected videos from {successful_selections}/{len(channel_results)} channels"
            )

        # Write JSON output if requested
        if args.out:
            if args.single_channel:
                # Single channel mode - simplified JSON
                channel_result = channel_results[0]
                if channel_result.picked:
                    # Extract video ID from URL
                    video_id = "N/A"
                    if "v=" in channel_result.picked.url:
                        video_id = channel_result.picked.url.split("v=")[1].split("&")[
                            0
                        ]

                    output_data = {
                        "channel_name": channel_result.channel_name,
                        "video_id": video_id,
                        "title": channel_result.picked.title,
                        "url": channel_result.picked.url,
                        "confidence": channel_result.confidence,
                        "published_at": channel_result.picked.published_at.isoformat(),
                        "published_at_formatted": channel_result.picked.published_at.strftime(
                            "%Y-%m-%d %H:%M UTC"
                        ),
                        "reasoning": channel_result.reasoning,
                        "matched_signals": channel_result.matched_signals,
                        "gameweek": result.gameweek,
                        "generated_at": result.generated_at.isoformat(),
                    }
                else:
                    output_data = {
                        "channel_name": channel_result.channel_name,
                        "video_id": None,
                        "title": None,
                        "url": None,
                        "confidence": 0.0,
                        "error": "No suitable video found",
                        "gameweek": result.gameweek,
                        "generated_at": result.generated_at.isoformat(),
                    }
            else:
                # Multi-channel mode - existing JSON format
                output_data = {
                    "channels": [
                        {
                            "channel_name": cr.channel_name,
                            "channel_url": cr.channel_url,
                            "picked": {
                                "title": cr.picked.title,
                                "url": cr.picked.url,
                                "published_at": cr.picked.published_at.isoformat(),
                                "description": cr.picked.description,
                            }
                            if cr.picked
                            else None,
                            "alternatives": [
                                {
                                    "title": video.title,
                                    "url": video.url,
                                    "published_at": video.published_at.isoformat(),
                                    "description": video.description,
                                }
                                for video in cr.alternatives
                            ],
                            "confidence": cr.confidence,
                            "reasoning": cr.reasoning,
                            "matched_signals": cr.matched_signals,
                            "heuristic_score": cr.heuristic_score,
                        }
                        for cr in channel_results
                    ],
                    "gameweek": result.gameweek,
                    "generated_at": result.generated_at.isoformat(),
                    "model": result.model,
                    "heuristic_notes": result.heuristic_notes,
                }

            with Path(args.out).open("w") as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Results written to {args.out}")

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


__all__ = [
    "ChannelResult",
    "SelectionResult",
    "VideoItem",
    "VideoPickerError",
    "cli_main",
    "select_single_channel",
    "setup_logging",
]
