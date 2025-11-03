#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "youtube-transcript-api==1.2.2",
# ]
# ///
"""
Simple test runner that doesn't conflict with project pytest configuration.
"""

import sys
from pathlib import Path

# Add current directory to path to import the main module
sys.path.insert(0, str(Path(__file__).parent))

from typing import TYPE_CHECKING

from fpl_transcript import format_as_txt, parse_video_id, srt_timestamp, vtt_timestamp

if TYPE_CHECKING:
    from fpl_influencer_hivemind.types import TranscriptSegment


def test_parse_video_id() -> None:
    """Test video ID parsing."""
    print("Testing parse_video_id...")

    # Test cases
    test_cases = [
        ("Y4qdYjBCyNc", "Y4qdYjBCyNc"),  # Raw ID
        ("https://www.youtube.com/watch?v=Y4qdYjBCyNc", "Y4qdYjBCyNc"),  # Watch URL
        ("https://youtu.be/Y4qdYjBCyNc", "Y4qdYjBCyNc"),  # Short URL
        ("https://www.youtube.com/embed/Y4qdYjBCyNc", "Y4qdYjBCyNc"),  # Embed
        ("https://www.youtube.com/shorts/Y4qdYjBCyNc", "Y4qdYjBCyNc"),  # Shorts
        ("https://youtu.be/Y4qdYjBCyNc?t=42", "Y4qdYjBCyNc"),  # With params
    ]

    for input_val, expected in test_cases:
        result = parse_video_id(input_val)
        assert result == expected, (
            f"Expected {expected}, got {result} for input {input_val}"
        )
        print(f"  ✓ {input_val[:50]}{'...' if len(input_val) > 50 else ''} -> {result}")

    print("✓ parse_video_id tests passed!\n")


def test_timestamps() -> None:
    """Test timestamp formatting."""
    print("Testing timestamp formatting...")

    # Test SRT timestamps
    assert srt_timestamp(0.0) == "00:00:00,000"
    assert srt_timestamp(125.5) == "00:02:05,500"
    assert srt_timestamp(3665.25) == "01:01:05,250"
    print("  ✓ SRT timestamps work correctly")

    # Test VTT timestamps
    assert vtt_timestamp(0.0) == "00:00:00.000"
    assert vtt_timestamp(125.5) == "00:02:05.500"
    assert vtt_timestamp(3665.25) == "01:01:05.250"
    print("  ✓ VTT timestamps work correctly")

    print("✓ Timestamp tests passed!\n")


def test_formatting() -> None:
    """Test transcript formatting functions."""
    print("Testing transcript formatting...")

    # Sample data
    sample_data: list[TranscriptSegment] = [
        {"text": "Hello there", "start": 0.0, "duration": 1.5},
        {"text": "How are you?", "start": 1.5, "duration": 2.3},
    ]

    # Test TXT format
    txt_result = format_as_txt(sample_data)
    assert "Hello there" in txt_result
    assert "How are you?" in txt_result
    print("  ✓ TXT formatting works")

    # Test TXT with timestamps
    txt_with_ts = format_as_txt(sample_data, include_timestamps=True)
    assert "[00:00:00]" in txt_with_ts
    print("  ✓ TXT with timestamps works")

    print("✓ Formatting tests passed!\n")


if __name__ == "__main__":
    try:
        test_parse_video_id()
        test_timestamps()
        test_formatting()
        print("🎉 All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
