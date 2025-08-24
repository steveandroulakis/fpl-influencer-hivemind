#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "youtube-transcript-api==1.2.2",
#   "pytest>=8.0.0",
# ]
# ///
"""
Unit tests for fpl_transcript.py

Tests focus on helper functions and formatting logic using synthetic data.
Network-dependent functionality is not tested here.

Run with: uv run test_transcript_fetcher.py
"""

import sys
from pathlib import Path

import pytest

# Add current directory to path to import the main module
sys.path.insert(0, str(Path(__file__).parent))

from fpl_transcript import (
    format_as_csv,
    format_as_json,
    format_as_srt,
    format_as_txt,
    format_as_vtt,
    parse_video_id,
    srt_timestamp,
    vtt_timestamp,
)


class TestParseVideoId:
    """Test video ID parsing from various URL formats."""

    def test_raw_video_id(self):
        """Test parsing raw 11-character video ID."""
        video_id = "Y4qdYjBCyNc"
        assert parse_video_id(video_id) == video_id

    def test_watch_url_basic(self):
        """Test basic YouTube watch URL."""
        url = "https://www.youtube.com/watch?v=Y4qdYjBCyNc"
        assert parse_video_id(url) == "Y4qdYjBCyNc"

    def test_watch_url_with_params(self):
        """Test watch URL with additional parameters."""
        url = "https://www.youtube.com/watch?v=Y4qdYjBCyNc&t=42s&si=abcd1234"
        assert parse_video_id(url) == "Y4qdYjBCyNc"

    def test_youtu_be_short_url(self):
        """Test youtu.be short URL format."""
        url = "https://youtu.be/Y4qdYjBCyNc"
        assert parse_video_id(url) == "Y4qdYjBCyNc"

    def test_youtu_be_with_timestamp(self):
        """Test youtu.be URL with timestamp parameter."""
        url = "https://youtu.be/Y4qdYjBCyNc?t=42"
        assert parse_video_id(url) == "Y4qdYjBCyNc"

    def test_embed_url(self):
        """Test YouTube embed URL format."""
        url = "https://www.youtube.com/embed/Y4qdYjBCyNc"
        assert parse_video_id(url) == "Y4qdYjBCyNc"

    def test_shorts_url(self):
        """Test YouTube Shorts URL format."""
        url = "https://www.youtube.com/shorts/Y4qdYjBCyNc"
        assert parse_video_id(url) == "Y4qdYjBCyNc"

    def test_mobile_url(self):
        """Test mobile YouTube URL format."""
        url = "https://m.youtube.com/watch?v=Y4qdYjBCyNc"
        assert parse_video_id(url) == "Y4qdYjBCyNc"

    def test_no_protocol(self):
        """Test URLs without protocol."""
        url = "www.youtube.com/watch?v=Y4qdYjBCyNc"
        assert parse_video_id(url) == "Y4qdYjBCyNc"

    def test_no_www(self):
        """Test URLs without www."""
        url = "https://youtube.com/watch?v=Y4qdYjBCyNc"
        assert parse_video_id(url) == "Y4qdYjBCyNc"

    def test_http_protocol(self):
        """Test HTTP (not HTTPS) URLs."""
        url = "http://www.youtube.com/watch?v=Y4qdYjBCyNc"
        assert parse_video_id(url) == "Y4qdYjBCyNc"

    def test_invalid_video_id_too_short(self):
        """Test error handling for invalid video ID (too short)."""
        with pytest.raises(
            ValueError, match="Could not extract valid YouTube video ID"
        ):
            parse_video_id("shortid")

    def test_invalid_video_id_too_long(self):
        """Test error handling for invalid video ID (too long)."""
        with pytest.raises(
            ValueError, match="Could not extract valid YouTube video ID"
        ):
            parse_video_id("toolongvideoid12345")

    def test_invalid_url_format(self):
        """Test error handling for completely invalid URL."""
        with pytest.raises(
            ValueError, match="Could not extract valid YouTube video ID"
        ):
            parse_video_id("https://example.com/not-a-youtube-url")

    def test_empty_string(self):
        """Test error handling for empty string."""
        with pytest.raises(
            ValueError, match="Could not extract valid YouTube video ID"
        ):
            parse_video_id("")


class TestTimestampFormatting:
    """Test timestamp formatting functions."""

    def test_srt_timestamp_zero(self):
        """Test SRT timestamp formatting for zero seconds."""
        assert srt_timestamp(0.0) == "00:00:00,000"

    def test_srt_timestamp_basic(self):
        """Test SRT timestamp formatting for basic time."""
        assert srt_timestamp(125.5) == "00:02:05,500"

    def test_srt_timestamp_hours(self):
        """Test SRT timestamp formatting with hours."""
        assert srt_timestamp(3665.250) == "01:01:05,250"

    def test_srt_timestamp_fractional_seconds(self):
        """Test SRT timestamp formatting with fractional seconds."""
        # Use a value that doesn't have floating point precision issues
        assert srt_timestamp(42.125) == "00:00:42,125"

    def test_vtt_timestamp_zero(self):
        """Test VTT timestamp formatting for zero seconds."""
        assert vtt_timestamp(0.0) == "00:00:00.000"

    def test_vtt_timestamp_basic(self):
        """Test VTT timestamp formatting for basic time."""
        assert vtt_timestamp(125.5) == "00:02:05.500"

    def test_vtt_timestamp_hours(self):
        """Test VTT timestamp formatting with hours."""
        assert vtt_timestamp(3665.250) == "01:01:05.250"

    def test_vtt_timestamp_fractional_seconds(self):
        """Test VTT timestamp formatting with fractional seconds."""
        # Use a value that doesn't have floating point precision issues
        assert vtt_timestamp(42.125) == "00:00:42.125"


class TestFormatting:
    """Test transcript formatting functions."""

    @pytest.fixture
    def sample_transcript_data(self):
        """Sample transcript data for testing."""
        return [
            {"text": "Hello there", "start": 0.0, "duration": 1.5},
            {"text": "How are you doing today?", "start": 1.5, "duration": 2.3},
            {"text": "This is a test", "start": 3.8, "duration": 1.8},
        ]

    def test_format_as_txt_basic(self, sample_transcript_data):
        """Test basic text formatting without timestamps."""
        result = format_as_txt(sample_transcript_data, include_timestamps=False)
        expected = "Hello there\nHow are you doing today?\nThis is a test"
        assert result == expected

    def test_format_as_txt_with_timestamps(self, sample_transcript_data):
        """Test text formatting with timestamps."""
        result = format_as_txt(sample_transcript_data, include_timestamps=True)
        lines = result.split("\n")

        assert len(lines) == 3
        assert lines[0].startswith("[00:00:00] Hello there")
        assert lines[1].startswith("[00:00:01] How are you doing today?")
        assert lines[2].startswith("[00:00:03] This is a test")

    def test_format_as_json(self, sample_transcript_data):
        """Test JSON formatting."""
        result = format_as_json(sample_transcript_data)

        # Should be valid JSON
        import json

        parsed = json.loads(result)

        assert len(parsed) == 3
        assert parsed[0]["text"] == "Hello there"
        assert parsed[0]["start"] == 0.0
        assert parsed[0]["duration"] == 1.5

    def test_format_as_csv(self, sample_transcript_data):
        """Test CSV formatting."""
        result = format_as_csv(sample_transcript_data)
        lines = result.strip().split("\n")

        # Should have header + 3 data rows
        assert len(lines) == 4
        assert "start,duration,text" in lines[0]  # Allow for carriage returns
        assert "0.0,1.5,Hello there" in lines[1]
        assert "1.5,2.3,How are you doing today?" in lines[2]
        assert "3.8,1.8,This is a test" in lines[3]

    def test_format_as_srt(self, sample_transcript_data):
        """Test SRT formatting."""
        result = format_as_srt(sample_transcript_data)
        lines = result.split("\n")

        # Check structure: number, timestamp, text, empty line for each cue
        assert "1" in lines  # First cue number
        assert "00:00:00,000 --> 00:00:01,500" in lines  # First timestamp
        assert "Hello there" in lines  # First text

        assert "2" in lines  # Second cue number
        assert "00:00:01,500 --> 00:00:03,800" in lines  # Second timestamp
        assert "How are you doing today?" in lines  # Second text

    def test_format_as_vtt(self, sample_transcript_data):
        """Test WebVTT formatting."""
        result = format_as_vtt(sample_transcript_data)
        lines = result.split("\n")

        # Should start with WebVTT header
        assert lines[0] == "WEBVTT"
        assert lines[1] == ""  # Empty line after header

        # Check for timestamp and text formatting
        assert "00:00:00.000 --> 00:00:01.500" in lines
        assert "Hello there" in lines
        assert "00:00:01.500 --> 00:00:03.800" in lines
        assert "How are you doing today?" in lines

    def test_format_empty_transcript(self):
        """Test formatting behavior with empty transcript."""
        empty_data = []

        assert format_as_txt(empty_data) == ""
        assert format_as_json(empty_data) == "[]"
        assert format_as_srt(empty_data) == ""
        assert format_as_vtt(empty_data) == "WEBVTT\n\n"

        # CSV should still have header
        csv_result = format_as_csv(empty_data)
        assert "start,duration,text" in csv_result

    def test_format_special_characters(self):
        """Test formatting with special characters in text."""
        special_data = [
            {"text": 'Text with "quotes" and <tags>', "start": 0.0, "duration": 1.0},
            {"text": "émojis 🎉 and ñoñó", "start": 1.0, "duration": 1.0},
        ]

        # Should preserve special characters
        txt_result = format_as_txt(special_data)
        assert '"quotes"' in txt_result
        assert "🎉" in txt_result
        assert "ñoñó" in txt_result

        # JSON should handle unicode properly
        json_result = format_as_json(special_data)
        assert "🎉" in json_result
        assert "ñoñó" in json_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
