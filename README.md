# FPL Influencer Hivemind

A comprehensive Fantasy Premier League (FPL) decision support system that aggregates data from popular FPL influencers and generates deterministic transfer and captain recommendations. LLMs are used only for transcript extraction and optional narrative summarization. The pipeline-first CLI collects influencer videos, transcripts, and FPL data to produce actionable recommendations for team `1178124` (or any entry you supply).

## 🚀 Quick Start
1. Install dependencies
   ```bash
   uv sync
   ```
2. Configure environment variables in `.env` (auto-loaded by the CLI – see `.env.example` for a template).
   ```bash
   export YOUTUBE_API_KEY="your-youtube-data-api-v3-key"     # Required for video discovery
   export ANTHROPIC_API_KEY="your-anthropic-api-key"        # Required for transcript extraction (and optional narrative summary)
   export YOUTUBE_TRANSCRIPT_IO_KEY="your-youtube-transcript-io-key"  # Primary transcript provider

   # Optional: fine-tune transcript behaviour
   export TRANSCRIPT_FETCHER_PREFERENCE="auto"               # Use "existing" to skip youtube-transcript.io
   export RAPIDAPI_EASYSUB_API_KEY="your-rapidapi-key"       # Optional fallback (EasySubAPI)
   export YOUTUBE_COOKIES_PATH="$HOME/youtube_cookies.txt"    # Optional fallback (yt-dlp + cookies)
   ```

### Getting API Keys
- **YouTube API Key**: Create a Google Cloud project, enable the YouTube Data API v3, and generate an API key from the **Credentials** section. The default daily quota (10,000 units) is plenty for routine runs.
- **YouTube Transcript IO Key**: Sign up at [youtube-transcript.io](https://www.youtube-transcript.io/), generate an API token, and set `YOUTUBE_TRANSCRIPT_IO_KEY`. This is now the default transcript provider (≈97× faster than the legacy flow).
- **RapidAPI EasySubAPI Key**: Optional fallback. Sign up at [RapidAPI](https://rapidapi.com/), subscribe to the [EasySubAPI](https://rapidapi.com/belchiorarkad-FqvHs2EDOtP/api/easysubapi) service, and copy the provided key if you want the legacy provider available.

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
  4. Interactive prompt lets you choose whether to run the deterministic analyzer (LLM extraction only).
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
- `src/fpl_influencer_hivemind/services/transcripts.py` – transcript gateway (YouTube Transcript IO by default, EasySubAPI/yt-dlp as fallback) returning text plus per-segment timing metadata.
- `src/fpl_influencer_hivemind/types.py` – shared TypedDict/dataclass models used across the pipeline, discovery, and CLI layers.
- `src/fpl_influencer_hivemind/analyzer/simple_orchestrator.py` – deterministic analyzer pipeline (LLM extraction only).
- `src/fpl_influencer_hivemind/youtube/video_picker.py` – reusable video discovery logic (also powers `youtube-titles/fpl_video_picker.py`).
- `src/fpl_influencer_hivemind/fpl/` – FPL API integration modules (gameweek, team, ownership data).
- `youtube-transcript/` – legacy transcript downloader retained for fallback and CLI experimentation.

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
- `./fpl_intelligence_analyzer.py --input var/hivemind/gwXX_teamXXXX_*.json --commentary "Triple captain this week"` to force a high-priority directive during analysis.
- `./fpl_intelligence_analyzer.py --input var/hivemind/gwXX_teamXXXX_*.json --narrative` to produce a narrative-only summary.

## 🧠 Deterministic Logic & Thresholds
- Consensus is computed by `element_id` from extracted mentions. Unresolved names are excluded from consensus counts.
- Gap severity = `(backers / total_channels) * 10`, with a +1 bump if `selected_by_percent >= 15`.
- Captain gap is triggered when the top consensus captain is not owned.
- Transfer options are deterministic: Roll (no transfers), Consensus (use available FTs), Conservative (1 FT when FTs > 1). Hits are not taken by default.
- Transfer constraints: position must match, budget must remain >= 0, club limit max 3, and incoming players must be in the top players list.
- Lineup scoring uses `ep_next`, then `form`, then `total_points` (scaled) to rank players and choose the best formation.
- Captain is the top consensus captain in the XI; otherwise the highest-scoring XI player. Vice is the next best XI player.
- Quality audit uses deterministic checks only (`validate_transfers`, `validate_lineup`).
- Update logic and thresholds in `src/fpl_influencer_hivemind/analyzer/simple_orchestrator.py`.

## 📁 Output & Logs
- Temporary working directories live under `var/hivemind/hivemind_*`.
- Final aggregation JSON and optional markdown reports are written to `var/hivemind/`.
- The pipeline prints emojis (`🔍`, `✅`, `❌`) to reflect per-channel discovery status.
- Aggregated transcripts now capture `{text, language, translated, segments[]}` so downstream tooling can preserve newline formatting and precise timing metadata.

The project is opinionated about a single entry ID but can target others by changing `--team-id`. Contributions should preserve the interactive CLI experience and the pipeline-first architecture described above.
