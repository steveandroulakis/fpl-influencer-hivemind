# FPL Influencer Hivemind

A comprehensive Fantasy Premier League (FPL) decision support system that aggregates data from popular FPL influencers and generates AI-powered transfer and captain recommendations using Claude-4 models. The pipeline-first CLI collects influencer videos, transcripts, and FPL data to produce actionable recommendations for team `1178124` (or any entry you supply).

## 🚀 Quick Start
1. Install dependencies
   ```bash
   uv sync
   ```
2. Configure environment variables in `.env` (auto-loaded by the CLI – see `.env.example` for a template).
   ```bash
   export YOUTUBE_API_KEY="your-youtube-data-api-v3-key"     # Required for video discovery
   export ANTHROPIC_API_KEY="your-anthropic-api-key"        # Required for LLM analysis

   # Optional: Alternative transcript fetching (bypasses IP blocking)
   export RAPIDAPI_EASYSUB_API_KEY="your-rapidapi-key"      # Optional for EasySubAPI transcript access
   export YOUTUBE_COOKIES_PATH="$HOME/youtube_cookies.txt"   # Optional for yt-dlp transcript access
   ```

### Getting API Keys
- **YouTube API Key**: Create a Google Cloud project, enable the YouTube Data API v3, and generate an API key from the **Credentials** section. The default daily quota (10,000 units) is plenty for routine runs.
- **RapidAPI EasySubAPI Key**: Sign up at [RapidAPI](https://rapidapi.com/), subscribe to the [EasySubAPI](https://rapidapi.com/belchiorarkad-FqvHs2EDOtP/api/easysubapi) service, and copy the provided key for transcript fetching.

### Optional: YouTube Cookies for yt-dlp
1. Install the Chrome extension [“Get cookies.txt LOCALLY”](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc).
2. Log in to YouTube in Chrome and visit [youtube.com](https://www.youtube.com/).
3. Use the extension → **Export** → save as `~/youtube_cookies.txt`.
4. Set `export YOUTUBE_COOKIES_PATH="$HOME/youtube_cookies.txt"` (or add to `.env`).

## 🎯 Primary Commands
- **Full pipeline**
  ```bash
  uv run fpl-influencer-hivemind pipeline --team-id 1178124 --free-transfers 2
  ```
  What happens:
  1. Gameweek + squad data gathered from the FPL API.
  2. Channel-by-channel video discovery runs in-process with live CLI updates.
  3. Transcript downloads execute (with progress per channel) and artifacts land in `var/hivemind/`.
  4. Interactive prompt lets you choose whether to run the Anthropic analyzer.
  Use `--commentary "Wildcard this week; recommend only wildcard route"` (or similar) to inject a user directive the analyzer treats as a primary requirement.

- **Data collection only**
  ```bash
  uv run fpl-influencer-hivemind collect --team-id 1178124 --auto-approve-transcripts
  ```
  Produces the same aggregation JSON without asking about analysis.

Artifacts are timestamped (e.g. `var/hivemind/gw05_team1178124_20250920T104530Z_aggregation.json`). Analyzer reports are written alongside the JSON when executed.

## 🧩 Architecture Essentials
- `src/fpl_influencer_hivemind/pipeline.py` – orchestration, CLI UX, transcript prompts, and logging callbacks.
- `src/fpl_influencer_hivemind/services/discovery.py` – pluggable channel discovery strategies (heuristics today, easy to extend).
- `src/fpl_influencer_hivemind/services/transcripts.py` – transcript gateway returning text plus per-segment timing metadata.
- `src/fpl_influencer_hivemind/types.py` – shared TypedDict/dataclass models used across the pipeline, discovery, and CLI layers.
- `src/fpl_influencer_hivemind/youtube/video_picker.py` – reusable video discovery logic (also powers `youtube-titles/fpl_video_picker.py`).
- `fpl/` – standalone scripts for FPL API data.
- `youtube-transcript/` – transcript downloader invoked by the pipeline.

All discovery now happens inside the main process, so the CLI remains responsive—no more silent subprocess wait.

## ✅ Quality Checklist
```bash
uv run ruff check src tests        # lint
uv run ruff format src tests       # format
uv run pytest                      # tests (>=80% coverage enforced)
```
Latest additions include unit coverage for the discovery helper (`tests/test_video_picker.py`) and updated pipeline tests that use stubs instead of spawning extra processes.

## 🔧 Useful Utilities
- `./youtube-titles/fpl_video_picker.py --single-channel "FPL Raptor" --gameweek 5 --verbose`
- `./youtube-transcript/fpl_transcript.py --id VIDEO_ID --format txt`
- `uv run fpl/get_my_team.py --entry-id 1178124 --show summary,picks`
- `./fpl_intelligence_analyzer.py --input var/hivemind/gwXX_teamXXXX_*.json --commentary "Triple captain this week"` to force a high-priority directive during analysis.

## 📁 Output & Logs
- Temporary working directories live under `var/hivemind/hivemind_*`.
- Final aggregation JSON and optional markdown reports are written to `var/hivemind/`.
- The pipeline prints emojis (`🔍`, `✅`, `❌`) to reflect per-channel discovery status.
- Aggregated transcripts now capture `{text, language, translated, segments[]}` so downstream tooling can preserve newline formatting and precise timing metadata.

The project is opinionated about a single entry ID but can target others by changing `--team-id`. Contributions should preserve the interactive CLI experience and the pipeline-first architecture described above.
