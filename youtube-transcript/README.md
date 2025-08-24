# YouTube Transcript Processing

This directory contains scripts for fetching and processing YouTube video transcripts, specifically designed for FPL (Fantasy Premier League) content analysis but suitable for any YouTube video with available transcripts.

## Scripts Overview

- **`fpl_transcript.py`** - Main transcript fetching tool with CLI interface
- **`test_transcript_fetcher.py`** - Comprehensive test suite 
- **`utils.py`** - Shared utilities for transcript processing
- **`requirements.txt`** - Dependencies for non-uv environments

## Quick Start

```bash
# Run directly with uv (recommended)
./fpl_transcript.py --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Or using uv run
uv run fpl_transcript.py --id VIDEO_ID --format json

# Run tests
uv run test_transcript_fetcher.py
```

## Main Tool: fpl_transcript.py

A robust command-line tool to fetch YouTube video transcripts using the official `youtube-transcript-api` library.

## Features

- **Multiple URL formats supported**: Full YouTube URLs, youtu.be short URLs, embed URLs, shorts, mobile URLs, or raw video IDs
- **Language handling**: Defaults to English with fallback to translation when available
- **5 output formats**: Plain text, JSON, CSV, SRT subtitles, WebVTT
- **Robust error handling**: Comprehensive exception handling with retry logic and exponential backoff
- **Flexible CLI**: Full control over languages, translation, output format, and retry behavior
- **Type hints & tests**: Well-tested code with comprehensive pytest suite

## Installation

### Requirements
- Python 3.10+
- Virtual environment recommended

### Setup

```bash
# Create and activate virtual environment
python -m venv transcript_env
source transcript_env/bin/activate  # On Windows: transcript_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Make script executable (optional)
chmod +x fpl_transcript.py
```

## Usage

### Basic Examples

```bash
# Fetch transcript as plain text
python fpl_transcript.py --url "https://www.youtube.com/watch?v=Y4qdYjBCyNc"

# Using video ID directly
python fpl_transcript.py --id Y4qdYjBCyNc --format json

# Save to file with timestamps
python fpl_transcript.py --url "https://youtu.be/Y4qdYjBCyNc" --format txt --include-timestamps --out transcript.txt

# Get SRT subtitles
python fpl_transcript.py --id Y4qdYjBCyNc --format srt --out subtitles.srt
```

### Advanced Usage

```bash
# Specify language preferences
python fpl_transcript.py --url "https://www.youtube.com/watch?v=Y4qdYjBCyNc" --languages en,es,fr

# Force translation to English
python fpl_transcript.py --id Y4qdYjBCyNc --translate-to en --verbose

# Custom retry behavior
python fpl_transcript.py --url "https://youtu.be/Y4qdYjBCyNc" --max-retries 5 --retry-backoff 2.0
```

## Command Line Options

### Required (choose one)
- `--url URL` - Full YouTube video URL
- `--id VIDEO_ID` - YouTube video ID (11 characters)

### Language Options
- `--languages LANGS` - Comma-separated preferred languages (default: `en,en-US,en-GB`)
- `--translate-to LANG` - Target language for translation fallback (default: `en`)

### Output Options
- `--format {txt,json,csv,srt,vtt}` - Output format (default: `txt`)
- `--include-timestamps` - Add timestamps to txt format
- `--out PATH` - Output file path (default: stdout)

### Retry Configuration
- `--max-retries INT` - Maximum retry attempts (default: 3)
- `--retry-backoff FLOAT` - Exponential backoff multiplier (default: 1.5)
- `--timeout FLOAT` - Request timeout seconds (default: 10.0)

### Other
- `--verbose` - Enable detailed logging
- `--help` - Show help message

## Output Formats

### TXT (Plain Text)
```
Hello everyone, FPraptor here and
welcome back to another video on my
YouTube channel. In today's video, we
```

### TXT with Timestamps
```
[00:00:00] Hello everyone, FPraptor here and
[00:00:01] welcome back to another video on my
[00:00:03] YouTube channel. In today's video, we
```

### JSON
```json
[
  {
    "text": "Hello everyone, FPraptor here and",
    "start": 0.0,
    "duration": 3.2
  },
  {
    "text": "welcome back to another video on my",
    "start": 1.6,
    "duration": 3.199
  }
]
```

### CSV
```csv
start,duration,text
0.0,3.2,"Hello everyone, FPraptor here and"
1.6,3.199,welcome back to another video on my
```

### SRT Subtitles
```
1
00:00:00,000 --> 00:00:03,200
Hello everyone, FPraptor here and

2
00:00:01,600 --> 00:00:04,799
welcome back to another video on my
```

### WebVTT
```
WEBVTT

00:00:00.000 --> 00:00:03.200
Hello everyone, FPraptor here and

00:00:01.600 --> 00:00:04.799
welcome back to another video on my
```

## Supported URL Formats

The script robustly handles various YouTube URL formats:

- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`  
- `https://www.youtube.com/shorts/VIDEO_ID`
- `https://m.youtube.com/watch?v=VIDEO_ID`
- URLs with additional parameters (`?t=42s`, `&si=abcd1234`)
- Raw 11-character video IDs

## Error Handling & Exit Codes

| Exit Code | Error Type | Description |
|-----------|------------|-------------|
| 0 | Success | Transcript retrieved successfully |
| 1 | Input Error | Invalid URL or video ID format |
| 2 | No Transcript | No transcript found for requested languages |
| 3 | Disabled | Transcripts disabled for this video |
| 4 | Unavailable | Video is unavailable or private |
| 5 | Blocked | Request blocked (rate limit/IP ban) |
| 6 | Retrieval Error | Generic transcript retrieval error |
| 7 | Unexpected | Unexpected error occurred |

## Troubleshooting

### Common Issues

#### "No transcript found"
- **Cause**: Video doesn't have transcripts in requested languages
- **Solution**: Try `--translate-to en` to attempt translation from available languages
- **Example**: `python fpl_transcript.py --id Y4qdYjBCyNc --languages en --translate-to en`

#### "Transcripts disabled"
- **Cause**: Video owner has disabled subtitle/transcript functionality
- **Solution**: No workaround available - try a different video

#### "Request blocked" or "IP blocked"
- **Cause**: YouTube has temporarily blocked your IP due to too many requests
- **Solutions**:
  1. Wait 10-15 minutes before trying again
  2. Use `--max-retries 5 --retry-backoff 2.0` for more patient retry behavior
  3. Try from a different network or use a VPN
- **Example**: `python fpl_transcript.py --id Y4qdYjBCyNc --max-retries 5 --verbose`

#### "Video unavailable"
- **Cause**: Video is private, deleted, or region-locked
- **Solution**: Verify the video URL is correct and publicly accessible

#### "Could not extract valid YouTube video ID"
- **Cause**: Invalid URL format or malformed video ID
- **Solution**: Verify URL is a valid YouTube link or provide 11-character video ID
- **Example**: Use `Y4qdYjBCyNc` instead of `https://invalid-url.com`

### Rate Limiting Best Practices

1. **Add delays between requests**: Use `time.sleep(1)` between multiple video fetches
2. **Use reasonable retry settings**: Default settings work for most cases
3. **Enable verbose logging**: Use `--verbose` to monitor request patterns
4. **Respect YouTube's terms**: Don't make excessive automated requests

## Development

### Running Tests
```bash
# Install test dependencies
pip install pytest

# Run test suite
python -m pytest test_fpl_transcript.py -v

# Run specific test
python -m pytest test_fpl_transcript.py::TestParseVideoId::test_watch_url_basic -v
```

### Test Coverage
The test suite covers:
- URL parsing for all supported formats
- Timestamp formatting (SRT and WebVTT)
- All output format generators
- Error handling for invalid inputs
- Edge cases and special characters

## Dependencies

- `youtube-transcript-api==1.2.2` - Official YouTube transcript API client
- `pytest>=8.0.0` - Testing framework (development only)

## License

This project is provided as-is for educational and personal use. Respect YouTube's Terms of Service and content creators' rights when using this tool.

## Next Steps

Potential enhancements for this tool:

1. **Language auto-detection**: Automatically detect video language before translation
2. **Keyword filtering**: Extract segments containing specific terms like "team selection"
3. **Multiple transcript support**: Save both native and translated transcripts
4. **Sentiment analysis**: Basic keyword/sentiment extraction from FPL content
5. **Batch processing**: Process multiple videos from a text file
6. **GUI interface**: Simple desktop app for non-technical users
7. **Database storage**: Store transcripts in SQLite for analysis
8. **FPL-specific parsing**: Extract player names, gameweek numbers, etc.

---

## Troubleshooting Examples

### Debug a problematic video
```bash
# Enable maximum verbosity and retries
python fpl_transcript.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --verbose --max-retries 5

# Try different language combinations
python fpl_transcript.py --id VIDEO_ID --languages en,en-US --translate-to en --verbose

# Test with minimal settings
python fpl_transcript.py --id VIDEO_ID --format json --verbose
```

### Batch processing script example
```bash
#!/bin/bash
# Process multiple FPL videos with error handling
videos=("Y4qdYjBCyNc" "ANOTHER_ID" "THIRD_ID")

for video_id in "${videos[@]}"; do
    echo "Processing $video_id..."
    python fpl_transcript.py --id "$video_id" --format json --out "transcript_${video_id}.json" --verbose
    if [ $? -eq 0 ]; then
        echo "✓ Success: $video_id"
    else
        echo "✗ Failed: $video_id"
    fi
    sleep 2  # Rate limiting
done
```

For more examples and advanced usage, see the comprehensive help:
```bash
python fpl_transcript.py --help
```