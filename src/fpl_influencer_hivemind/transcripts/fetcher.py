"""
FPL Transcript Fetcher

A robust command-line tool to fetch YouTube video transcripts using yt-dlp or EasySubAPI.
Supports multiple output formats (txt, json, csv, srt, vtt) with comprehensive error handling, retry logic, and authentication.

Methods:
- yt-dlp: Downloads subtitles directly from YouTube (supports cookies for IP block bypass)
- EasySubAPI: Alternative API service to bypass IP blocking (requires RAPIDAPI_EASYSUB_API_KEY)
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import tempfile
import time
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests
import yt_dlp  # type: ignore[import-untyped]
from yt_dlp.utils import DownloadError  # type: ignore[import-untyped]


def parse_video_id(url_or_id: str) -> str:
    """
    Extract YouTube video ID from various URL formats or return the ID if already provided.

    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - https://m.youtube.com/watch?v=VIDEO_ID
    - Raw video ID

    Args:
        url_or_id: YouTube URL or video ID

    Returns:
        str: YouTube video ID

    Raises:
        ValueError: If no valid video ID can be extracted
    """
    # If it's already an 11-character video ID, return it
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url_or_id):
        return url_or_id

    # Parse URL patterns
    patterns = [
        # Standard watch URL
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        # Short URL
        r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})",
        # Embed URL
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
        # Shorts URL
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
        # Mobile URL
        r"(?:https?://)?m\.youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    # Try parsing as URL with query parameters
    try:
        parsed = urlparse(url_or_id)
        if "v" in parse_qs(parsed.query):
            video_id = parse_qs(parsed.query)["v"][0]
            if re.match(r"^[a-zA-Z0-9_-]{11}$", video_id):
                return video_id
    except Exception:
        pass

    raise ValueError(f"Could not extract valid YouTube video ID from: {url_or_id}")


def srt_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        str: Formatted timestamp in SRT format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = round((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def vtt_timestamp(seconds: float) -> str:
    """
    Convert seconds to WebVTT timestamp format (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds

    Returns:
        str: Formatted timestamp in WebVTT format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = round((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def format_as_txt(
    transcript_data: list[dict[str, Any]], include_timestamps: bool = False
) -> str:
    """
    Format transcript as plain text.

    Args:
        transcript_data: List of transcript segments
        include_timestamps: Whether to include timestamp prefixes

    Returns:
        str: Formatted text
    """
    lines = []
    for segment in transcript_data:
        text = segment["text"].strip()
        if include_timestamps:
            timestamp = vtt_timestamp(segment["start"])
            timestamp_display = timestamp.split(".")[
                0
            ]  # Remove milliseconds for display
            lines.append(f"[{timestamp_display}] {text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def format_as_json(transcript_data: list[dict[str, Any]]) -> str:
    """
    Format transcript as JSON.

    Args:
        transcript_data: List of transcript segments

    Returns:
        str: JSON formatted string
    """
    return json.dumps(transcript_data, indent=2, ensure_ascii=False)


def format_as_csv(transcript_data: list[dict[str, Any]]) -> str:
    """
    Format transcript as CSV.

    Args:
        transcript_data: List of transcript segments

    Returns:
        str: CSV formatted string
    """
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=["start", "duration", "text"])
    writer.writeheader()
    writer.writerows(transcript_data)
    return output.getvalue()


def format_as_srt(transcript_data: list[dict[str, Any]]) -> str:
    """
    Format transcript as SRT.

    Args:
        transcript_data: List of transcript segments

    Returns:
        str: SRT formatted string
    """
    lines = []
    for i, segment in enumerate(transcript_data, 1):
        start_time = srt_timestamp(segment["start"])
        end_time = srt_timestamp(segment["start"] + segment["duration"])

        lines.extend(
            [
                str(i),
                f"{start_time} --> {end_time}",
                segment["text"].strip(),
                "",  # Empty line separator
            ]
        )

    return "\n".join(lines)


def format_as_vtt(transcript_data: list[dict[str, Any]]) -> str:
    """
    Format transcript as WebVTT.

    Args:
        transcript_data: List of transcript segments

    Returns:
        str: WebVTT formatted string
    """
    if not transcript_data:
        return "WEBVTT\n\n"

    lines = ["WEBVTT", ""]  # WebVTT header

    for segment in transcript_data:
        start_time = vtt_timestamp(segment["start"])
        end_time = vtt_timestamp(segment["start"] + segment["duration"])

        lines.extend(
            [
                f"{start_time} --> {end_time}",
                segment["text"].strip(),
                "",  # Empty line separator
            ]
        )

    return "\n".join(lines)


class YtDlpTranscriptFetcher:
    """Main class for fetching and formatting YouTube transcripts using yt-dlp."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        timeout: float = 10.0,
        cookies_path: str | None = None,
        delay: float = 0.0,
        random_delay: bool = False,
    ):
        """
        Initialize the transcript fetcher.

        Args:
            max_retries: Maximum number of retry attempts
            retry_backoff: Backoff multiplier for retries
            timeout: Timeout per request in seconds
            cookies_path: Path to YouTube cookies.txt file for bypassing IP blocks
            delay: Base delay between requests in seconds
            random_delay: Add random jitter to delays (±25%)
        """
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout
        self.cookies_path = cookies_path
        self.delay = delay
        self.random_delay = random_delay

    def _apply_delay(self, attempt: int = 0) -> None:
        """Apply configurable delay with optional jitter."""
        if self.delay <= 0:
            return

        delay = self.delay
        if attempt > 0:
            # Exponential backoff for retries
            delay = self.delay * (self.retry_backoff**attempt)

        if self.random_delay:
            # Add ±25% jitter
            jitter = random.uniform(-0.25, 0.25)
            delay = delay * (1 + jitter)

        logging.info(f"Applying delay: {delay:.2f}s")
        time.sleep(delay)

    def fetch_transcript(
        self,
        video_id: str,
        languages: list[str],
        translate_to: str = "en",  # noqa: ARG002
    ) -> tuple[list[dict[str, Any]], str, bool]:
        """
        Fetch transcript using yt-dlp with retry logic and cookie support.

        Args:
            video_id: YouTube video ID
            languages: List of preferred languages
            translate_to: Target language for translation if needed (not currently used)

        Returns:
            Tuple of (transcript_data, language_used, was_translated)

        Raises:
            Various yt-dlp exceptions
        """
        last_exception: Exception | None = None
        url = f"https://www.youtube.com/watch?v={video_id}"

        for attempt in range(self.max_retries):
            try:
                logging.info(
                    f"Attempt {attempt + 1}: Fetching transcript for video {video_id}"
                )

                # Apply delay before request (except first attempt)
                if attempt > 0:
                    self._apply_delay(attempt)

                # Configure yt-dlp options
                ydl_opts: dict[str, Any] = {
                    "writeautomaticsub": True,
                    "writesubtitles": True,
                    "subtitleslangs": languages,
                    "subtitlesformat": "vtt",
                    "skip_download": True,
                    "quiet": not logging.getLogger().isEnabledFor(logging.INFO),
                    "no_warnings": not logging.getLogger().isEnabledFor(logging.DEBUG),
                }

                # Add cookies if available
                if self.cookies_path and Path(self.cookies_path).exists():
                    ydl_opts["cookiefile"] = self.cookies_path
                    logging.debug(f"Using cookies from: {self.cookies_path}")

                # Create temporary directory for subtitle files
                with tempfile.TemporaryDirectory() as temp_dir:
                    ydl_opts["outtmpl"] = str(Path(temp_dir) / "%(id)s.%(ext)s")

                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        try:
                            ydl.extract_info(url, download=True)

                            # Find downloaded subtitle files
                            subtitle_files: list[tuple[str, str, bool]] = []
                            temp_path = Path(temp_dir)
                            for lang in languages:
                                for ext in ["vtt", "srt"]:
                                    subtitle_file = (
                                        temp_path / f"{video_id}.{lang}.{ext}"
                                    )
                                    if subtitle_file.exists():
                                        subtitle_files.append(
                                            (str(subtitle_file), lang, False)
                                        )
                                        break

                            # If no manual subtitles, look for auto-generated
                            if not subtitle_files:
                                for lang in languages:
                                    for ext in ["vtt", "srt"]:
                                        auto_file = (
                                            temp_path / f"{video_id}.{lang}.auto.{ext}"
                                        )
                                        if auto_file.exists():
                                            subtitle_files.append(
                                                (str(auto_file), lang, True)
                                            )
                                            break

                            if not subtitle_files:
                                # Look for any available subtitles
                                all_subs = list(temp_path.glob(f"{video_id}.*vtt"))
                                all_subs.extend(temp_path.glob(f"{video_id}.*srt"))
                                if all_subs:
                                    file_path = all_subs[0]
                                    # Extract language from filename (e.g., video_id.en.vtt)
                                    parts = file_path.name.split(".")
                                    lang = parts[1] if len(parts) >= 3 else "unknown"
                                    is_auto = "auto" in parts
                                    subtitle_files.append(
                                        (str(file_path), lang, is_auto)
                                    )

                            if subtitle_files:
                                subtitle_file_str: str
                                subtitle_file_str, language_used, was_auto = (
                                    subtitle_files[0]
                                )
                                transcript_data = self._parse_subtitle_file(
                                    subtitle_file_str
                                )
                                logging.info(
                                    f"Successfully extracted {len(transcript_data)} transcript segments"
                                )
                                return transcript_data, language_used, was_auto
                            else:
                                raise Exception("No subtitle files found")

                        except DownloadError as e:
                            error_msg = str(e).lower()
                            if any(
                                keyword in error_msg
                                for keyword in [
                                    "blocked",
                                    "429",
                                    "rate limit",
                                    "too many",
                                ]
                            ):
                                logging.warning(f"Rate limited/blocked: {e}")
                                raise Exception(f"Rate limited: {e}") from e
                            else:
                                raise Exception(f"Download error: {e}") from e

            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Check if it's a retriable error
                is_retriable = any(
                    keyword in error_msg
                    for keyword in [
                        "blocked",
                        "429",
                        "rate limit",
                        "too many",
                        "timeout",
                        "connection",
                    ]
                )

                if attempt < self.max_retries - 1 and is_retriable:
                    wait_time = (self.retry_backoff**attempt) * (
                        1 + 0.1 * random.random()
                    )
                    logging.warning(
                        f"Retriable error (attempt {attempt + 1}), retrying in {wait_time:.2f}s: {e}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Log the final error
                    if "blocked" in error_msg or "429" in error_msg:
                        logging.error(f"Request blocked/rate limited: {e}")
                        logging.error(
                            "This is expected if your IP is currently blocked"
                        )
                    else:
                        logging.error(f"Failed to fetch transcript: {e}")
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Failed to fetch transcript after all retries")

    def _parse_subtitle_file(self, file_path: str) -> list[dict[str, Any]]:
        """Parse VTT or SRT subtitle file into transcript format."""
        with Path(file_path).open(encoding="utf-8") as f:
            content = f.read()

        if file_path.endswith(".vtt"):
            return self._parse_vtt_content(content)
        elif file_path.endswith(".srt"):
            return self._parse_srt_content(content)
        else:
            raise ValueError(f"Unsupported subtitle format: {file_path}")

    def _parse_vtt_content(self, content: str) -> list[dict[str, Any]]:
        """Parse WebVTT content into transcript format."""
        transcript_data = []
        lines = content.strip().split("\n")
        i = 0

        # Skip WEBVTT header and initial empty lines
        while i < len(lines) and (not lines[i].strip() or "WEBVTT" in lines[i]):
            i += 1

        while i < len(lines):
            line = lines[i].strip()

            # Look for timestamp line (format: 00:00:01.000 --> 00:00:05.000)
            if " --> " in line:
                try:
                    start_str, end_str = line.split(" --> ")
                    start_time = self._parse_vtt_timestamp(start_str)
                    end_time = self._parse_vtt_timestamp(end_str)

                    i += 1
                    # Collect text lines until empty line or next timestamp
                    text_lines = []
                    while (
                        i < len(lines) and lines[i].strip() and " --> " not in lines[i]
                    ):
                        text_lines.append(lines[i].strip())
                        i += 1

                    if text_lines:
                        text = " ".join(text_lines)
                        # Remove VTT formatting tags
                        text = re.sub(r"<[^>]+>", "", text)
                        transcript_data.append(
                            {
                                "start": start_time,
                                "duration": end_time - start_time,
                                "text": text,
                            }
                        )
                except (ValueError, IndexError):
                    pass

            i += 1

        return transcript_data

    def _parse_srt_content(self, content: str) -> list[dict[str, Any]]:
        """Parse SRT content into transcript format."""
        transcript_data = []
        blocks = content.strip().split("\n\n")

        for block in blocks:
            lines = block.split("\n")
            if len(lines) >= 3:
                # lines[0] = sequence number
                # lines[1] = timestamp
                # lines[2:] = text
                try:
                    timestamp_line = lines[1]
                    start_str, end_str = timestamp_line.split(" --> ")
                    start_time = self._parse_srt_timestamp(start_str)
                    end_time = self._parse_srt_timestamp(end_str)

                    text = " ".join(lines[2:])
                    transcript_data.append(
                        {
                            "start": start_time,
                            "duration": end_time - start_time,
                            "text": text,
                        }
                    )
                except (ValueError, IndexError):
                    continue

        return transcript_data

    def _parse_vtt_timestamp(self, timestamp: str) -> float:
        """Parse VTT timestamp (HH:MM:SS.mmm) to seconds."""
        parts = timestamp.strip().split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
            total_seconds = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            total_seconds = float(minutes) * 60 + float(seconds)
        else:
            total_seconds = float(parts[0])
        return total_seconds

    def _parse_srt_timestamp(self, timestamp: str) -> float:
        """Parse SRT timestamp (HH:MM:SS,mmm) to seconds."""
        timestamp = timestamp.replace(",", ".")  # SRT uses comma for milliseconds
        return self._parse_vtt_timestamp(timestamp)


class EasySubApiFetcher:
    """Alternative transcript fetcher using EasySubAPI (RapidAPI) to bypass IP blocking."""

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        timeout: float = 10.0,
    ):
        """
        Initialize the EasySubAPI transcript fetcher.

        Args:
            api_key: RapidAPI EasySubAPI key
            max_retries: Maximum number of retry attempts
            retry_backoff: Backoff multiplier for retries
            timeout: Timeout per request in seconds
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout

    def fetch_transcript(
        self,
        video_id: str,
        languages: list[str],  # noqa: ARG002
        translate_to: str = "en",  # noqa: ARG002
    ) -> tuple[list[dict[str, Any]], str, bool]:
        """
        Fetch transcript using EasySubAPI with retry logic.

        Args:
            video_id: YouTube video ID
            languages: List of preferred languages (unused for EasySubAPI, always returns English)
            translate_to: Target language (unused for EasySubAPI)

        Returns:
            Tuple of (transcript_data, language_used, was_translated)

        Raises:
            Exception: If transcript fetching fails
        """
        url = "https://easysubapi.p.rapidapi.com/api/easysubapi-get-transcript"
        headers = {
            "Content-Type": "application/json",
            "x-rapidapi-host": "easysubapi.p.rapidapi.com",
            "x-rapidapi-key": self.api_key,
        }
        payload = {"video_id": video_id}

        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                logging.info(
                    f"Attempt {attempt + 1}: Fetching transcript via EasySubAPI for video {video_id}"
                )

                if attempt > 0:
                    wait_time = (self.retry_backoff**attempt) * (
                        1 + 0.1 * random.random()
                    )
                    logging.info(f"Waiting {wait_time:.2f}s before retry")
                    time.sleep(wait_time)

                response = requests.post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
                response.raise_for_status()

                data = response.json()

                if "result" not in data or not data["result"]:
                    raise Exception("No results found in API response")

                # Extract transcript frames from nested structure
                result_item = data["result"][0]
                if "data" not in result_item or "frames" not in result_item["data"]:
                    raise Exception("Invalid API response structure")

                frames = result_item["data"]["frames"]
                if not frames:
                    raise Exception("No transcript frames found")

                # Transform to standard format
                transcript_data = self._transform_easysub_data(frames)

                logging.info(
                    f"Successfully extracted {len(transcript_data)} transcript segments via EasySubAPI"
                )
                return transcript_data, "en", False

            except requests.exceptions.RequestException as e:
                last_exception = e
                error_msg = str(e).lower()

                is_retriable = any(
                    keyword in error_msg
                    for keyword in [
                        "timeout",
                        "connection",
                        "429",
                        "rate limit",
                        "too many",
                        "server error",
                    ]
                )

                if attempt < self.max_retries - 1 and is_retriable:
                    logging.warning(f"Retriable error (attempt {attempt + 1}): {e}")
                    continue
                else:
                    logging.error(f"EasySubAPI request failed: {e}")
                    raise Exception(f"EasySubAPI error: {e}") from e

            except Exception as e:
                last_exception = e
                logging.error(f"EasySubAPI processing failed: {e}")

                if attempt < self.max_retries - 1:
                    logging.warning(f"Retrying (attempt {attempt + 1})")
                    continue
                else:
                    raise

        if last_exception:
            raise last_exception
        raise RuntimeError("Failed to fetch transcript after all retries")

    def _transform_easysub_data(
        self, frames: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Transform EasySubAPI frames to standard transcript format.

        Args:
            frames: List of frames from EasySubAPI

        Returns:
            List of transcript segments in standard format
        """
        transcript_data = []

        for frame in frames:
            try:
                start_time = float(frame["from_time"])
                end_time = float(frame["to_time"])
                text = frame["message"].strip()

                if text:  # Skip empty segments
                    transcript_data.append(
                        {
                            "start": start_time,
                            "duration": end_time - start_time,
                            "text": text,
                        }
                    )

            except (KeyError, ValueError, TypeError) as e:
                logging.warning(f"Skipping malformed frame: {frame} - {e}")
                continue

        return transcript_data


def create_transcript_fetcher(
    rapidapi_key: str | None = None,
    cookies_path: str | None = None,
    max_retries: int = 3,
    retry_backoff: float = 1.5,
    timeout: float = 10.0,
    delay: float = 0.0,
    random_delay: bool = False,
    api_method: str = "auto",
) -> YtDlpTranscriptFetcher | EasySubApiFetcher:
    """
    Factory function to create appropriate transcript fetcher based on configuration.

    Args:
        rapidapi_key: EasySubAPI key (if available)
        cookies_path: Path to YouTube cookies file
        max_retries: Maximum retry attempts
        retry_backoff: Backoff multiplier
        timeout: Request timeout
        delay: Delay between requests (yt-dlp only)
        random_delay: Add random jitter (yt-dlp only)
        api_method: Method to use ('auto', 'easysub', 'ytdlp')

    Returns:
        Configured transcript fetcher instance
    """
    if api_method == "easysub":
        if not rapidapi_key:
            raise ValueError("EasySubAPI method requires RAPIDAPI_EASYSUB_API_KEY")
        logging.info("Using EasySubAPI for transcript fetching")
        return EasySubApiFetcher(rapidapi_key, max_retries, retry_backoff, timeout)

    elif api_method == "ytdlp":
        logging.info("Using yt-dlp for transcript fetching")
        return YtDlpTranscriptFetcher(
            max_retries, retry_backoff, timeout, cookies_path, delay, random_delay
        )

    else:  # auto
        if rapidapi_key:
            logging.info("EasySubAPI key detected - using EasySubAPI as primary method")
            return EasySubApiFetcher(rapidapi_key, max_retries, retry_backoff, timeout)
        else:
            logging.info("Using yt-dlp for transcript fetching")
            return YtDlpTranscriptFetcher(
                max_retries, retry_backoff, timeout, cookies_path, delay, random_delay
            )


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch YouTube video transcripts in multiple formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --url https://www.youtube.com/watch?v=VIDEO_ID
  %(prog)s --id VIDEO_ID --format json --out transcript.json
  %(prog)s --url https://youtu.be/VIDEO_ID --format srt --include-timestamps
  %(prog)s --id VIDEO_ID --languages en,es --translate-to en --verbose
  %(prog)s --id VIDEO_ID --api-method easysub --verbose

Environment Variables:
  RAPIDAPI_EASYSUB_API_KEY    RapidAPI key for EasySubAPI (enables IP block bypass)
  YOUTUBE_COOKIES_PATH        Path to YouTube cookies.txt for yt-dlp authentication
        """,
    )

    # URL/ID input (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--url", help="YouTube video URL")
    input_group.add_argument("--id", help="YouTube video ID")

    # Language options
    parser.add_argument(
        "--languages",
        default="en,en-US,en-GB",
        help="Comma-separated list of preferred language codes (default: en,en-US,en-GB)",
    )
    parser.add_argument(
        "--translate-to",
        default="en",
        help="Target language for translation if native language not available (default: en)",
    )

    # Output format options
    parser.add_argument(
        "--format",
        choices=["txt", "json", "csv", "srt", "vtt"],
        default="txt",
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "--include-timestamps",
        action="store_true",
        help="Include timestamps in txt format (ignored for other formats)",
    )
    parser.add_argument("--out", help="Output file path (default: stdout)")

    # Retry configuration
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts (default: 3)",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.5,
        help="Backoff multiplier for retries (default: 1.5)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Timeout per request in seconds (default: 10.0)",
    )

    # Authentication
    parser.add_argument(
        "--cookies",
        help="Path to YouTube cookies.txt file (overrides YOUTUBE_COOKIES_PATH env var)",
    )

    # API method selection
    parser.add_argument(
        "--api-method",
        choices=["auto", "easysub", "ytdlp"],
        default="auto",
        help="Transcript fetching method (default: auto - use EasySubAPI if key available, else yt-dlp)",
    )

    # Rate limiting
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between requests in seconds (default: 0.0, recommended: 10-20 for rate limiting)",
    )
    parser.add_argument(
        "--random-delay",
        action="store_true",
        help="Add random jitter (±25%%) to delays to avoid detection patterns",
    )

    # Logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level, format="%(levelname)s: %(message)s", stream=sys.stderr
    )

    try:
        # Extract video ID
        video_input = args.url or args.id
        video_id = parse_video_id(video_input)
        logging.info(f"Extracted video ID: {video_id}")

        # Parse languages
        languages = [lang.strip() for lang in args.languages.split(",")]
        logging.info(f"Language preference: {languages}")

        # Determine cookies path (CLI arg overrides env var)
        cookies_path = args.cookies or os.environ.get("YOUTUBE_COOKIES_PATH")
        if cookies_path:
            logging.info(f"Using cookies from: {cookies_path}")
        else:
            logging.info(
                "No cookies specified for yt-dlp - using unauthenticated requests"
            )

        # Get RapidAPI key
        rapidapi_key = os.environ.get("RAPIDAPI_EASYSUB_API_KEY")

        # Create fetcher using factory function
        fetcher = create_transcript_fetcher(
            rapidapi_key=rapidapi_key,
            cookies_path=cookies_path,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            timeout=args.timeout,
            delay=args.delay,
            random_delay=args.random_delay,
            api_method=args.api_method,
        )

        transcript_data, language_used, was_translated = fetcher.fetch_transcript(
            video_id, languages, args.translate_to
        )

        if was_translated:
            logging.info(f"Transcript translated to {language_used}")
        else:
            logging.info(f"Transcript retrieved in {language_used}")

        # Format output
        formatters = {
            "txt": lambda data: format_as_txt(data, args.include_timestamps),
            "json": format_as_json,
            "csv": format_as_csv,
            "srt": format_as_srt,
            "vtt": format_as_vtt,
        }

        formatted_output = formatters[args.format](transcript_data)

        # Output
        if args.out:
            Path(args.out).write_text(formatted_output, encoding="utf-8")
            logging.info(f"Output written to: {args.out}")
        else:
            print(formatted_output, end="")

    except ValueError as e:
        logging.error(f"Input error: {e}")
        sys.exit(1)
    except Exception as e:
        error_msg = str(e).lower()

        if "no subtitle" in error_msg or "no transcript" in error_msg:
            logging.error(f"No transcript found: {e}")
            sys.exit(2)
        elif "blocked" in error_msg or "429" in error_msg or "rate limit" in error_msg:
            logging.error(f"Request blocked/rate limited: {e}")
            logging.error("This is expected if your IP is currently blocked")
            logging.error("Try again later, use different network, or use VPN/proxy")
            sys.exit(5)
        elif "unavailable" in error_msg:
            logging.error(f"Video unavailable: {e}")
            sys.exit(4)
        else:
            logging.error(f"Unexpected error: {e}")
            if args.verbose:
                raise
            sys.exit(7)


if __name__ == "__main__":
    main()
