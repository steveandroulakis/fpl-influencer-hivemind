#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "youtube-transcript-api==1.2.2",
# ]
# ///
"""
FPL Transcript Fetcher

A robust command-line tool to fetch YouTube video transcripts using the youtube-transcript-api library.
Supports multiple output formats (txt, json, csv, srt, vtt) with comprehensive error handling and retry logic.
"""

import argparse
import csv
import json
import logging
import re
import sys
import time
from io import StringIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    NotTranslatable,
    RequestBlocked,
    TranscriptsDisabled,
    TranslationLanguageNotAvailable,
    VideoUnavailable,
    YouTubeRequestFailed,
)


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


def format_as_txt(transcript_data: list[dict], include_timestamps: bool = False) -> str:
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


def format_as_json(transcript_data: list[dict]) -> str:
    """
    Format transcript as JSON.

    Args:
        transcript_data: List of transcript segments

    Returns:
        str: JSON formatted string
    """
    return json.dumps(transcript_data, indent=2, ensure_ascii=False)


def format_as_csv(transcript_data: list[dict]) -> str:
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


def format_as_srt(transcript_data: list[dict]) -> str:
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


def format_as_vtt(transcript_data: list[dict]) -> str:
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


class TranscriptFetcher:
    """Main class for fetching and formatting YouTube transcripts."""

    def __init__(
        self, max_retries: int = 3, retry_backoff: float = 1.5, timeout: float = 10.0
    ):
        """
        Initialize the transcript fetcher.

        Args:
            max_retries: Maximum number of retry attempts
            retry_backoff: Backoff multiplier for retries
            timeout: Timeout per request in seconds
        """
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.timeout = timeout
        self.api = YouTubeTranscriptApi()

    def fetch_transcript(
        self, video_id: str, languages: list[str], translate_to: str = "en"
    ) -> tuple[list[dict], str, bool]:
        """
        Fetch transcript with retry logic and translation fallback.

        Args:
            video_id: YouTube video ID
            languages: List of preferred languages
            translate_to: Target language for translation if needed

        Returns:
            Tuple of (transcript_data, language_used, was_translated)

        Raises:
            Various youtube_transcript_api exceptions
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                logging.info(
                    f"Attempt {attempt + 1}: Fetching transcript for video {video_id}"
                )

                # Try direct fetch with preferred languages
                try:
                    transcript = self.api.fetch(video_id, languages=languages)
                    return transcript.to_raw_data(), transcript.language_code, False
                except NoTranscriptFound:
                    # If direct fetch fails, try finding available transcripts and translating
                    if translate_to in languages:
                        logging.info("Direct fetch failed, attempting translation...")
                        return self._fetch_with_translation(
                            video_id, languages, translate_to
                        )
                    else:
                        raise

            except (RequestBlocked, IpBlocked, YouTubeRequestFailed) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = (self.retry_backoff**attempt) * (
                        1 + 0.1 * time.time() % 1
                    )  # Add jitter
                    logging.warning(
                        f"Request failed (attempt {attempt + 1}), retrying in {wait_time:.2f}s: {e}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    raise
            except (
                TranscriptsDisabled,
                VideoUnavailable,
                NotTranslatable,
                TranslationLanguageNotAvailable,
            ):
                # These are not retriable errors
                raise
            except CouldNotRetrieveTranscript:
                # Generic transcript retrieval error
                raise
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = (self.retry_backoff**attempt) * (
                        1 + 0.1 * time.time() % 1
                    )
                    logging.warning(
                        f"Unexpected error (attempt {attempt + 1}), retrying in {wait_time:.2f}s: {e}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    raise

        if last_exception:
            raise last_exception

    def _fetch_with_translation(
        self, video_id: str, languages: list[str], translate_to: str
    ) -> tuple[list[dict], str, bool]:
        """
        Attempt to fetch transcript with translation.

        Args:
            video_id: YouTube video ID
            languages: Preferred languages
            translate_to: Target translation language

        Returns:
            Tuple of (transcript_data, language_used, was_translated)
        """
        transcript_list = self.api.list(video_id)

        # Try to find any available transcript and translate it
        for transcript in transcript_list:
            if transcript.is_translatable:
                try:
                    translated = transcript.translate(translate_to).fetch()
                    return translated.to_raw_data(), translate_to, True
                except (NotTranslatable, TranslationLanguageNotAvailable):
                    continue

        # If translation fails, try any available transcript in original language
        try:
            transcript = transcript_list.find_transcript(languages)
            fetched = transcript.fetch()
            return fetched.to_raw_data(), fetched.language_code, False
        except NoTranscriptFound:
            # Last resort: get any available transcript
            if len(transcript_list) > 0:
                first_transcript = next(iter(transcript_list))
                fetched = first_transcript.fetch()
                return fetched.to_raw_data(), fetched.language_code, False
            else:
                raise NoTranscriptFound(video_id, languages, transcript_list) from None


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

        # Create fetcher and get transcript
        fetcher = TranscriptFetcher(
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            timeout=args.timeout,
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
    except NoTranscriptFound as e:
        logging.error(f"No transcript found: {e}")
        sys.exit(2)
    except TranscriptsDisabled as e:
        logging.error(f"Transcripts disabled: {e}")
        sys.exit(3)
    except VideoUnavailable as e:
        logging.error(f"Video unavailable: {e}")
        sys.exit(4)
    except (RequestBlocked, IpBlocked) as e:
        logging.error(f"Request blocked: {e}")
        logging.error("Try again later or use a VPN/proxy")
        sys.exit(5)
    except CouldNotRetrieveTranscript as e:
        logging.error(f"Could not retrieve transcript: {e}")
        sys.exit(6)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if args.verbose:
            raise
        sys.exit(7)


if __name__ == "__main__":
    main()
