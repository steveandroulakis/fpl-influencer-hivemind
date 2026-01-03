# CLAUDE.md

## Project Snapshot
- Purpose: collect FPL influencer sentiment, transcripts, and recommendations for entry `1178124`.
- Runtime: Python 3.11+ via `uv` virtual environment; `.env` (see `.env.example`) is auto-loaded by the pipeline.
- Description: comprehensive FPL decision-support that pairs data collection with Claude-4 analysis for transfer/captain picks.
- CLI: single entrypoint `uv run fpl-influencer-hivemind` with `collect` and `pipeline` subcommands.

## Core Flow
1. **FPL data** (`src/fpl_influencer_hivemind/fpl/`) – fetched in-process via package modules, no manual steps.
2. **YouTube discovery** (`src/fpl_influencer_hivemind/services/discovery.py`) – pluggable strategies that wrap the heuristic `video_picker` helper.
3. **Transcript fetch** (`src/fpl_influencer_hivemind/services/transcripts.py`) – yields newline-preserving text plus segment timing metadata (YouTube Transcript IO by default, yt-dlp/EasySubAPI fallback).
4. **Analysis** (`fpl_intelligence_analyzer.py`) – optional, requires `ANTHROPIC_API_KEY`.

Artifacts are written under `var/hivemind/` with timestamped filenames.

## Analyzer Stages
The analyzer (`fpl_intelligence_analyzer.py`) runs a multi-stage pipeline with validation:

### Stage 1: Gap Analysis
- Identifies gaps between user's squad and influencer consensus
- Outputs: `players_to_sell`, `players_missing`, `risk_flags`, `captain_gap`

### Stage 2: Transfer Plan
- Generates specific transfers addressing gaps
- Validates: budget, club limits, position matching
- **Cohesion validation**: checks gaps are addressed or justified in reasoning
- **Consensus validation**: flags if 3+ influencer recs ignored (warning) or 4+ majority ignored (error)
- Retries up to 2x on validation failure

### Stage 3: Lineup Selection
- Selects starting XI, bench order, captain/vice
- Validates: 11 starters, 4 bench, formation rules (1 GKP, 3-5 DEF, 2-5 MID, 1-3 FWD)
- **Risk validation**: risky captain needs safe vice; risky XI players need bench backup
- Retries up to 2x on validation failure

### Stage 4: Validation
- Combines mechanical checks (budget, formation) + cohesion checks (stage consistency)
- Uses LLM (haiku) to verify justifications in reasoning text
- Errors trigger retry; warnings are logged but don't block

### Stage 5: Quality Review (Corrective)
- Holistic LLM review (sonnet) for **internal consistency only** (no model knowledge)
- Compares: gap analysis vs transfers vs lineup vs influencer consensus
- Outputs `QualityReview` with `fixable_issues` array
- **Corrective loop**: if fixable issues found, re-runs affected stages with fix instructions
- Re-runs quality review after corrections
- Final report includes "Quality Assessment" section with remaining observations

### Key Models (`types.py`)
- `GapAnalysis`, `TransferPlan`, `LineupPlan` – stage outputs
- `ValidationResult` – errors/warnings from validation
- `QualityReview` – holistic assessment with `fixable_issues: list[FixableIssue]`
- `FixableIssue` – issue + stage + fix_instruction for corrective loop

## Required Environment
Use `.env` (auto sourced) based on `.env.example`:
- `YOUTUBE_API_KEY` – YouTube Data API v3 key (needed for video discovery).
- `YOUTUBE_TRANSCRIPT_IO_KEY` – API token for youtube-transcript.io (primary transcript provider).
- `ANTHROPIC_API_KEY` – Claude API key (needed for analyzer, optional otherwise).
- Optional: `RAPIDAPI_EASYSUB_API_KEY`, `YOUTUBE_COOKIES_PATH` for transcript fallbacks.
- Optional: `TRANSCRIPT_FETCHER_PREFERENCE` (set to `existing` to bypass youtube-transcript.io or `auto` for default).
- `PATH` additions for local `uv`/scripts are appended automatically.

### Credential Tips
- **YouTube API Key**: create a Google Cloud project → enable YouTube Data API v3 → generate an API key (10k default quota is sufficient).
- **YouTube Transcript IO**: sign up at youtube-transcript.io → generate an API key → store as `YOUTUBE_TRANSCRIPT_IO_KEY`.
- **RapidAPI EasySubAPI**: sign up at RapidAPI → subscribe to EasySubAPI → copy the key to keep the legacy fallback available.
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
- When editing docs, keep README and CLAUDE aligned with the "pipeline-first" architecture.

## Type Safety Rules
Write code that passes mypy strict mode, Pylance, and Ruff. Follow these patterns:

### Import Type Annotations
- Always import `Any` from `typing` when needed for dynamic/external data.
- Use `# type: ignore[import-untyped]` for third-party libraries without stubs (e.g., `yt_dlp`, `googleapiclient`).
- Install stub packages when available (e.g., `types-requests` for `requests`).
- Import specific exception types from untyped libraries with type ignore comments.

### Function Signatures
- **All functions must have complete type annotations** for parameters and return types.
- Use `-> None` explicitly for functions that don't return values.
- Annotate callback/mock parameters in tests with proper signatures.
- Use `noqa: ARG001` for intentionally unused arguments (e.g., in stubs or interface implementations).

### Type Definitions
- **TypedDict with mixed required/optional fields**: split into base required class, then inherit with `total=False`:
  ```python
  class _MyTypeRequired(TypedDict):
      required_field: str
  
  class MyType(_MyTypeRequired, total=False):
      optional_field: NotRequired[str]
  ```
- Use modern union syntax: `str | int | None` instead of `Union[str, int, None]` or `Optional[str]`.
- Use `type[Any]` instead of `Type[Any]` (PEP 585).

### Variable Annotations
- **Always annotate empty containers** and variables with non-obvious types:
  ```python
  results: dict[str, tuple[VideoItem, Response]] = {}
  subtitle_files: list[tuple[str, str, bool]] = []
  last_exception: Exception | None = None
  ```
- Annotate variables when type inference may be ambiguous (e.g., `handler: logging.Handler = logging.StreamHandler()`).

### Type Narrowing & Assertions
- **Global variables**: type checkers can't narrow globals after None checks—use `cast()`:
  ```python
  if _FPL_CLASS is None:
      _FPL_CLASS = module.FPL
  return cast(type[Any], _FPL_CLASS)
  ```
- **API response validation**: check structure and narrow types explicitly before use:
  ```python
  if not message.content or not hasattr(message.content[0], "text"):
      raise ValueError("Expected text response")
  if isinstance(message.content[0], TextBlock):
      response_text = message.content[0].text
  ```
- **Parser return types**: validate that parsers return expected types (e.g., `dateutil.parser.parse` can return `date` or `datetime`).

### Collections & Paths
- Use `Path` from `pathlib` for file operations instead of string paths and `os.path`.
- Use `.exists()`, `.glob()`, etc. on `Path` objects instead of `os.path.exists()` and `glob.glob()`.
- Prefer `isinstance(obj, str | int | float | bool | type(None))` over multiple `isinstance` calls.

### External Data
- Use `dict[str, Any]` for JSON/API responses with unknown structure.
- Use `list[dict[str, Any]]` for collections of unstructured data.
- Add `# type: ignore[no-any-return]` when returning `Any` from typed functions (only when unavoidable).

### Test Fixtures
- Mock objects and test doubles need full type signatures matching real implementations.
- Use `TYPE_CHECKING` guard for test-only type imports to avoid runtime dependencies.
- Annotate fixture return types explicitly.

### Configuration
The project uses strict mypy settings in `pyproject.toml`:
- `strict = true` with `warn_return_any` and `warn_unused_configs`.
- `mypy_path = "src"` and `explicit_package_bases = true` for proper package resolution.
- `ignore_missing_imports = false` to catch all import issues (add type ignore comments as needed).

## File Guide
- `fpl_intelligence_analyzer.py` – wrapper script calling `analyzer.cli.main()`.
- `src/fpl_influencer_hivemind/analyzer/` – refactored multi-stage LLM analyzer module:
  - `__init__.py` – public exports (`FPLIntelligenceAnalyzer`, `ChannelAnalysis`).
  - `constants.py` – `PL_TEAMS_2025_26`, `PL_TEAMS_CONTEXT`.
  - `models.py` – `ChannelAnalysis`, `PlayerLookupEntry`, `SquadPlayerEntry`, `DecisionOption`.
  - `api.py` – `AnthropicClient`, `make_anthropic_call` with retry, `extract_last_json`.
  - `normalization.py` – name helpers (`normalize_name`, `canonicalize_player_label`, `build_player_lookup`).
  - `stages/gap.py` – `stage_gap_analysis`, `aggregate_influencer_consensus`.
  - `stages/transfer.py` – `stage_transfer_plan`, `apply_transfer_pricing`, `compute_post_transfer_squad`.
  - `stages/lineup.py` – `stage_lineup_selection`, `aggregate_influencer_xi`.
  - `validation/cohesion.py` – `validate_gap_to_transfer_cohesion`, `validate_consensus_coverage`, `validate_risk_contingency`.
  - `validation/mechanical.py` – `validate_transfers`, `validate_lineup`, `validate_all`.
  - `quality.py` – `holistic_quality_review`.
  - `report.py` – `generate_consensus_section`, `generate_channel_notes`, `format_gap_section`, `format_action_plan`, `assemble_report`.
  - `orchestrator.py` – `FPLIntelligenceAnalyzer` class, `_run_staged_analysis`, `_build_decision_options`.
  - `cli.py` – `main()`, argparse.
- `src/fpl_influencer_hivemind/pipeline.py` – orchestrator + logging callbacks + transcript prompts.
- `src/fpl_influencer_hivemind/services/discovery.py` – strategy layer for channel discovery (heuristic default).
- `src/fpl_influencer_hivemind/services/transcripts.py` – transcript wrapper returning `{text, language, translated, segments[]}`.
- `src/fpl_influencer_hivemind/types.py` – shared TypedDict/dataclass/Pydantic models (includes analyzer stage outputs).
- `src/fpl_influencer_hivemind/youtube/video_picker.py` – reusable discovery logic + CLI shim (`youtube-titles/fpl_video_picker.py`).
- `tests/test_video_picker.py` – unit tests for discovery module.
- `tests/test_pipeline.py` – end-to-end aggregation behaviour under stubs.

## Before You Finish
1. Run `uv run ruff check src tests`.
2. Run `uv run pytest` (ensures ≥80% coverage).
3. Verify artifacts land in `var/hivemind/` and CLI output remains interactive.
