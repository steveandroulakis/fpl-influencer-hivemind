"""
Shared utilities for YouTube transcript processing.

This module contains common functions that may be reused across multiple
transcript-related scripts in the future.
"""

import re
from urllib.parse import parse_qs, urlparse


def parse_video_id_from_url(url_or_id: str) -> str:
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


def clean_transcript_text(text: str) -> str:
    """
    Clean and normalize transcript text.

    Args:
        text: Raw transcript text

    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    # Strip leading/trailing whitespace
    return text.strip()


def filter_transcript_by_keywords(
    transcript_data: list[dict], keywords: list[str], case_sensitive: bool = False
) -> list[dict]:
    """
    Filter transcript segments containing specific keywords.

    Args:
        transcript_data: List of transcript segments
        keywords: List of keywords to search for
        case_sensitive: Whether to perform case-sensitive search

    Returns:
        List[Dict]: Filtered transcript segments
    """
    if not keywords:
        return transcript_data

    filtered_segments = []

    for segment in transcript_data:
        text = segment["text"]
        if not case_sensitive:
            text = text.lower()
            search_keywords = [k.lower() for k in keywords]
        else:
            search_keywords = keywords

        # Check if any keyword appears in this segment
        if any(keyword in text for keyword in search_keywords):
            filtered_segments.append(segment)

    return filtered_segments


def get_transcript_duration(transcript_data: list[dict]) -> float:
    """
    Calculate total duration of transcript.

    Args:
        transcript_data: List of transcript segments

    Returns:
        float: Total duration in seconds
    """
    if not transcript_data:
        return 0.0

    # Find the end time of the last segment
    last_segment = transcript_data[-1]
    return last_segment["start"] + last_segment["duration"]


def summarize_transcript_stats(transcript_data: list[dict]) -> dict:
    """
    Generate basic statistics about a transcript.

    Args:
        transcript_data: List of transcript segments

    Returns:
        Dict: Statistics including word count, segment count, duration
    """
    if not transcript_data:
        return {
            "total_segments": 0,
            "total_words": 0,
            "total_duration": 0.0,
            "average_segment_duration": 0.0,
            "words_per_minute": 0.0,
        }

    total_segments = len(transcript_data)
    total_words = sum(len(segment["text"].split()) for segment in transcript_data)
    total_duration = get_transcript_duration(transcript_data)
    average_segment_duration = (
        total_duration / total_segments if total_segments > 0 else 0.0
    )
    words_per_minute = (
        (total_words / (total_duration / 60)) if total_duration > 0 else 0.0
    )

    return {
        "total_segments": total_segments,
        "total_words": total_words,
        "total_duration": total_duration,
        "average_segment_duration": average_segment_duration,
        "words_per_minute": words_per_minute,
    }
