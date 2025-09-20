# CLAUDE.md

## Project Snapshot
- Purpose: collect FPL influencer sentiment, transcripts, and recommendations for entry `1178124`.
- Runtime: Python 3.11+ via `uv` virtual environment; `.env` (see `.env.example`) is auto-loaded by the pipeline.
- Description: comprehensive FPL decision-support that pairs data collection with Claude-4 analysis for transfer/captain picks.
- CLI: single entrypoint `uv run fpl-influencer-hivemind` with `collect` and `pipeline` subcommands.

## Core Flow
1. **FPL data** (`fpl/` scripts) – fetched in-process, no manual steps.
2. **YouTube discovery** (`src/fpl_influencer_hivemind/youtube/video_picker.py`) – in-process logging, Anthropic ranking fallback to heuristics.
3. **Transcript fetch** (`youtube-transcript/fpl_transcript.py`) – still invoked as a subprocess, now with progress logs per channel.
4. **Analysis** (`fpl_intelligence_analyzer.py`) – optional, requires `ANTHROPIC_API_KEY`.

Artifacts are written under `var/hivemind/` with timestamped filenames.

## Required Environment
Use `.env` (auto sourced) based on `.env.example`:
- `YOUTUBE_API_KEY` – YouTube Data API v3 key (needed for video discovery).
- `ANTHROPIC_API_KEY` – Claude API key (needed for analyzer, optional otherwise).
- Optional: `RAPIDAPI_EASYSUB_API_KEY`, `YOUTUBE_COOKIES_PATH` for transcript fallbacks.
- `PATH` additions for local `uv`/scripts are appended automatically.

### Credential Tips
- **YouTube API Key**: create a Google Cloud project → enable YouTube Data API v3 → generate an API key (10k default quota is sufficient).
- **RapidAPI EasySubAPI**: sign up at RapidAPI → subscribe to EasySubAPI → copy the key for reliable transcripts.
- **YouTube cookies**: install “Get cookies.txt LOCALLY”, export cookies while logged into YouTube, save to `~/youtube_cookies.txt`, and set `YOUTUBE_COOKIES_PATH`.

## Everyday Commands
- `uv sync` – install dependencies (lock file managed).
- `uv run fpl-influencer-hivemind pipeline --team-id 1178124 --free-transfers 2` – full run.
- `uv run fpl-influencer-hivemind collect --team-id 1178124 --auto-approve-transcripts` – data only.
- `uv run pytest` – full test suite with coverage threshold (80%).
- `uv run ruff check src tests` / `uv run ruff format src tests` – lint & format.

## Development Tips
- Prefer importing modules (`select_single_channel`, etc.) instead of shelling out.
- Tests use helper stubs (`stub_select_single_channel`) to avoid real API calls; keep them in sync with pipeline behaviour.
- Coverage pragmas exist for API-heavy classes—only add new ones when external services are unavoidable.
- When editing docs, keep README and CLAUDE aligned with the “pipeline-first” architecture.

## File Guide
- `src/fpl_influencer_hivemind/pipeline.py` – orchestrator and CLI UX (progress prints, transcript prompts).
- `src/fpl_influencer_hivemind/youtube/video_picker.py` – reusable discovery logic + CLI shim (`youtube-titles/fpl_video_picker.py`).
- `tests/test_video_picker.py` – unit tests for discovery module.
- `tests/test_pipeline.py` – end-to-end aggregation behaviour under stubs.

## Before You Finish
1. Run `uv run ruff check src tests`.
2. Run `uv run pytest` (ensures ≥80% coverage).
3. Verify artifacts land in `var/hivemind/` and CLI output remains interactive.
