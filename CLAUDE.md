# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project that scrapes popular Fantasy Premier League (FPL) influencers to analyze their upcoming gameweek opinions and generate recommendations. The project uses modern Python tooling (uv, ruff, mypy) and follows strict type checking and testing practices.

## Development Commands

### Setup and Installation
- `uv sync` - Install dependencies and sync environment
- `uv add package-name` - Add new runtime dependencies  
- `uv add --dev package-name` - Add new development dependencies

### Code Quality and Testing
- `uv run pytest` - Run tests with coverage (must reach 80% minimum)
- `uv run pytest tests/test_specific.py` - Run specific test file
- `uv run pytest tests/test_specific.py::test_function` - Run specific test
- `uv run ruff check .` - Lint code (fix issues with `--fix`)
- `uv run ruff format .` - Format code
- `uv run mypy src` - Type check (strict mode, no `Any` types allowed)

### Automation with Nox
- `nox` - Run all quality checks across Python versions (3.11, 3.12, 3.13)
- `nox -s tests` - Run tests across Python versions
- `nox -s lint` - Run linting checks
- `nox -s typecheck` - Run type checking
- `nox -s coverage` - Generate HTML and XML coverage reports

### Running Scripts and Application
- `uv run fpl-influencer-hivemind` - Run the main application
- `uv run python -m fpl_influencer_hivemind` - Alternative way to run main application
- `uv run script_name.py` - Run individual scripts with project dependencies
- `./script_name.py` - Run scripts directly (if executable and using PEP 723 shebang)

### Script-Specific Commands
#### Main Pipeline - Complete FPL Analysis
- `./fpl_data_aggregator.sh --team-id 1178124 --output-file results.json` - **Full pipeline orchestrator**
- `./fpl_data_aggregator.sh --team-id 1178124 --verbose` - Complete analysis with detailed logging
- `./fpl_intelligence_analyzer.py --input results.json --output-file analysis.md --verbose` - **LLM-powered analysis and recommendations**

#### FPL Data Analysis (`./fpl/`)
- `uv run fpl/get_team_players.py --team "Arsenal" --format json` - Club roster analysis
- `uv run fpl/get_top_ownership.py --limit 200 --out top_players.csv` - Ownership rankings
- `uv run fpl/get_current_gameweek.py --date 2025-08-24` - Gameweek information
- `uv run fpl/get_my_team.py --entry-id 1178124 --show "summary,picks"` - Personal team analysis

#### YouTube Video Processing (`./youtube-titles/`)
- `export YOUTUBE_API_KEY="your-key"` - **Required** for YouTube Data API v3 access
- `export ANTHROPIC_API_KEY="your-key"` - Required for Anthropic integration
- `./youtube-titles/fpl_video_picker.py --gameweek 5 --max-per-channel 3 --days 30` - Find FPL team selection videos (all channels)
- `./youtube-titles/fpl_video_picker.py --single-channel "FPL Raptor" --gameweek 2 --out raptor.json` - Single channel processing
- `./youtube-titles/run_parallel_channels.sh --gameweek 2 --days 7 --output results.json` - **Parallel processing all channels**
- `./youtube-titles/run_parallel_channels.sh --gameweek 3 --verbose` - Parallel with detailed logging
- **Uses YouTube Data API v3** for reliable data extraction and **channels.json** for configuration

#### YouTube Transcript Processing (`./youtube-transcript/`)
- `./youtube-transcript/fpl_transcript.py --url "https://www.youtube.com/watch?v=VIDEO_ID"` - Auto-select API method (EasySubAPI if key available)
- `uv run youtube-transcript/fpl_transcript.py --id VIDEO_ID --format json --out transcript.json` - Export as JSON
- `./youtube-transcript/fpl_transcript.py --id VIDEO_ID --api-method easysub --verbose` - Force EasySubAPI method
- `./youtube-transcript/fpl_transcript.py --id VIDEO_ID --api-method ytdlp --cookies ~/youtube_cookies.txt` - Force yt-dlp with cookies
- `uv run youtube-transcript/fpl_transcript.py --url "https://youtu.be/VIDEO_ID" --format srt --include-timestamps` - Generate SRT subtitles
- **Dual API support**: yt-dlp (direct YouTube) or EasySubAPI (IP-blocking resistant)
- **Supports 5 formats**: txt, json, csv, srt, vtt with robust error handling and retry logic
- **Cookie Support**: yt-dlp method uses `YOUTUBE_COOKIES_PATH` env var or `--cookies` flag for authentication

## Code Architecture

### Project Structure
- `src/fpl_influencer_hivemind/` - Main package following modern src-layout
- `tests/` - Test suite with pytest configuration
- Root-level directories (`./youtube-transcript`, `./youtube-titles`, `./fpl`, etc.) - Isolated development scripts
- Package is installable and has CLI entry point configured

### Development Approach
The project follows a **script-first development model**:
1. **Isolated scripts** in root directories for experimentation and focused development
2. **Shared dependencies** managed at project level through pyproject.toml
3. **Eventual integration** of successful scripts into main package under `/src`

### Script Development
- Use **PEP 723 inline metadata** for standalone scripts with specific dependencies:
  ```python
  #!/usr/bin/env -S uv run --script
  # /// script
  # requires-python = ">=3.11"
  # dependencies = ["requests", "beautifulsoup4"]
  # ///
  ```
- Run scripts directly: `./script_name.py` or `uv run script_name.py`
- Add common dependencies to main pyproject.toml as patterns emerge

### External API Integration Patterns
- **FPL API**: Uses official FPL endpoints (`https://fantasy.premierleague.com/api/`) with `fpl` library
- **YouTube Data API v3**: Production integration for video metadata extraction
  - Requires `YOUTUBE_API_KEY` environment variable
  - Handles channel URL resolution (@handles, channel IDs, custom URLs)
  - Uses uploads playlists with pagination for efficient video fetching
- **Anthropic Claude**: Production integration for intelligent content analysis with JSON schema validation
  - Requires `ANTHROPIC_API_KEY` environment variable
  - Uses structured prompts with fallback strategies
  - Implements robust JSON parsing with error recovery
  - **Supports both batched and individual processing modes** for optimal API usage

### Parallel Processing Architecture
The project implements a **shell-script orchestrated parallel processing model** for scalable YouTube analysis:
- **Single-channel mode**: `--single-channel "FPL Raptor"` processes one channel independently
- **Shell orchestrator**: `run_parallel_channels.sh` launches multiple single-channel processes in parallel
- **Channel configuration**: `channels.json` provides structured channel metadata for 5 FPL influencers
- **Result collation**: Individual JSON results are combined into unified output with summary statistics
- **Fault tolerance**: One channel failure doesn't break the entire pipeline
- **Asyncio ready**: Clean video_id + channel_name extraction optimized for future Python async integration

### FPL Integration Context
The project is designed to work with Fantasy Premier League data and includes:
- **Team ID**: 1178124 (configured in guide-fpl-data.md)
- **FPL API**: Uses `https://fantasy.premierleague.com/api/` endpoints
- **Recommended library**: `fpl` package (https://github.com/amosbastian/fpl)
- **Data processing**: Expected to use pandas for player/team data analysis

### Development Standards
- **Type hints**: All code must use strict type hints (mypy --strict)
- **Test coverage**: Minimum 80% coverage required, configured to fail below threshold
- **Code style**: Ruff replaces black/flake8/isort with comprehensive rule set
- **Python versions**: Supports 3.11+ (configured for 3.11, 3.12, 3.13)

## Key Configuration Details

### Tool Configuration
- Ruff: 88 character line length, comprehensive rule set including pyupgrade, bugbear, comprehensions
- MyPy: Strict mode with warn_return_any and warn_unused_configs enabled
- Pytest: Runs with coverage reporting, uses src path for imports
- Pre-commit: Configured for ruff, mypy, and standard hooks

### Coverage Exclusions
Coverage excludes common patterns like `__repr__`, debug statements, `NotImplementedError`, and abstract methods to focus on meaningful test coverage.

## Typical Workflow

### Complete FPL Analysis Pipeline
1. **Data Collection**: `./fpl_data_aggregator.sh --team-id 1178124 --output-file analysis_data.json --verbose`
   - Collects current gameweek info, top 150 player data, personal team analysis
   - Discovers relevant FPL videos from 5 YouTube channels in parallel
   - Fetches and cleans transcripts for LLM processing
   - Outputs comprehensive JSON with all aggregated data

2. **LLM Analysis**: `./fpl_intelligence_analyzer.py --input analysis_data.json --output-file gameweek_X_analysis.md --verbose`
   - Phase 1: Sonnet-4 analyzes each channel individually for team selections, transfers, captain choices
   - Phase 2: Opus-4 generates comparative analysis with specific recommendations for your team
   - Considers player injury status (status codes: a/d/i/s/u) in all recommendations
   - Outputs 160+ line markdown report with executive summary, transfer recommendations, captain analysis

### Individual Component Testing
- Test FPL API: `uv run fpl/get_my_team.py --entry-id 1178124`
- Test YouTube API setup: `./youtube-titles/test_ytapi.py` - Validates YOUTUBE_API_KEY and channel resolution
- Test video discovery: `./youtube-titles/fpl_video_picker.py --single-channel "FPL Raptor" --gameweek 5 --verbose`
- Test transcripts: `./youtube-transcript/fpl_transcript.py --id VIDEO_ID --format txt`
- Test EasySubAPI: `./youtube-transcript/fpl_transcript.py --id VIDEO_ID --api-method easysub --verbose`

### Environment Variables
Required for full functionality:
- `YOUTUBE_API_KEY` - YouTube Data API v3 key for video metadata extraction (required for youtube-titles/)
- `ANTHROPIC_API_KEY` - Anthropic Claude API key for intelligent video analysis (optional, fallback to heuristics)
- `RAPIDAPI_EASYSUB_API_KEY` - RapidAPI EasySubAPI key for IP-blocking-resistant transcript fetching (optional)
- `YOUTUBE_COOKIES_PATH` - Path to YouTube cookies.txt file for bypassing IP blocks via yt-dlp (optional, e.g. `~/youtube_cookies.txt`)

## Current Implementation Status

### Completed Components
- **Complete Analysis Pipeline**: End-to-end FPL decision support system
  - `fpl_data_aggregator.sh` - **Main orchestrator** combining all data collection in parallel
  - `fpl_intelligence_analyzer.py` - **LLM-powered analysis engine** using Claude-4 models for recommendations
  - **Two-phase LLM processing**: Individual channel analysis + comprehensive comparative analysis
  - **Injury/availability integration**: Player status codes (a/d/i/s/u) factored into all recommendations
  - **Markdown report generation**: 160+ line detailed analysis with actionable insights
- **FPL API Integration** (`./fpl/`): Production-ready scripts for team analysis, ownership data, and gameweek management
  - `get_team_players.py` - Club roster analysis with fuzzy team name matching
  - `get_top_ownership.py` - Most owned players ranked by selection percentage  
  - `get_current_gameweek.py` - Smart gameweek detection with timezone handling
  - `get_my_team.py` - Personal FPL team analysis using entry ID 1178124
  - Shared utilities in `utils.py` for API access and data formatting
- **YouTube Video Processing** (`./youtube-titles/`): Intelligent FPL content discovery with parallel processing
  - `fpl_video_picker.py` - Production-ready script using YouTube Data API v3 + Anthropic Claude
  - **Dual processing modes**: Multi-channel batch processing OR single-channel mode (`--single-channel`)
  - `run_parallel_channels.sh` - **Shell orchestrator for true parallel processing** across all channels
  - `channels.json` - **Structured channel configuration** with names, URLs, and descriptions
  - Handles all YouTube URL formats (@handles, channel IDs, custom URLs) with automatic resolution
  - Heuristic filtering with keyword scoring for "team selection" content
  - Anthropic API integration for intelligent video ranking with JSON schema validation
  - **Optimized for asyncio integration** with clean video_id + channel_name extraction
  - Comprehensive error handling, fault tolerance, and fallback strategies
- **YouTube Transcript Processing** (`./youtube-transcript/`): Robust transcript fetching and formatting
  - `fpl_transcript.py` - Production-ready CLI tool with dual API support
  - **Dual API architecture**: yt-dlp (direct YouTube) or EasySubAPI (IP-blocking resistant)
  - **Auto-selection logic**: Uses EasySubAPI when `RAPIDAPI_EASYSUB_API_KEY` available, fallback to yt-dlp
  - Multiple URL format support (watch, youtu.be, embed, shorts, mobile, raw IDs)
  - 5 output formats (txt, json, csv, srt, vtt) with proper timestamp formatting
  - Language handling with English preference and translation fallback
  - **Dual authentication support**: cookies for yt-dlp, RapidAPI key for EasySubAPI
  - Comprehensive error handling with retry logic and exponential backoff
  - Factory pattern for clean API method selection and backwards compatibility

### Development Pipeline
Scripts demonstrate full production readiness with:
- PEP 723 inline dependencies for isolated execution
- Multiple output formats (table, JSON, CSV) 
- Comprehensive CLI interfaces with argparse
- Robust error handling and user feedback
- Type hints throughout with strict mypy compliance

### Key Output Formats
- **Single-channel JSON**: `{"channel_name": "FPL Raptor", "video_id": "Y4qdYjBCyNc", "confidence": 0.95}`
- **Parallel processing JSON**: `{"channels": [...], "summary": {"successful_channels": 5}}`
- **Multi-channel batch JSON**: Unified results from all channels with alternatives and detailed metadata
- **Console output**: Human-readable summaries with confidence scores and reasoning from Anthropic Claude

### Migration Architecture
Successful scripts follow this integration pattern:
```
./script_directory/               →    src/fpl_influencer_hivemind/
  ├── script_name.py             →      ├── component_name/
  ├── utils.py                   →      │   ├── core_logic.py
  ├── requirements.txt           →      │   └── api_utils.py  
  └── README.md                  →      └── cli/unified_interface.py
```