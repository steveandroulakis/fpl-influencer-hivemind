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
- `src/fpl_influencer_hivemind/pipeline.py` – orchestrator + logging callbacks + transcript prompts.
- `src/fpl_influencer_hivemind/services/discovery.py` – strategy layer for channel discovery (heuristic default).
- `src/fpl_influencer_hivemind/services/transcripts.py` – transcript wrapper returning `{text, language, translated, segments[]}`.
- `src/fpl_influencer_hivemind/types.py` – shared TypedDict/dataclass definitions for discovery/transcripts/gameweek data.
- `src/fpl_influencer_hivemind/youtube/video_picker.py` – reusable discovery logic + CLI shim (`youtube-titles/fpl_video_picker.py`).
- `tests/test_video_picker.py` – unit tests for discovery module.
- `tests/test_pipeline.py` – end-to-end aggregation behaviour under stubs.

## Before You Finish
1. Run `uv run ruff check src tests`.
2. Run `uv run pytest` (ensures ≥80% coverage).
3. Verify artifacts land in `var/hivemind/` and CLI output remains interactive.
