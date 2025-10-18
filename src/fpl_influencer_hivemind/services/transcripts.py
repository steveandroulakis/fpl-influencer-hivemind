"""Transcript retrieval helpers."""

from __future__ import annotations

import importlib
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..types import TranscriptEntry, TranscriptSegment

_FETCHER_MODULE = "fpl_influencer_hivemind.transcripts.fetcher"
_FETCHER_CACHE: tuple[object, object] | None = None


class TranscriptServiceError(RuntimeError):
    """Raised when transcript retrieval fails."""


def _load_fetcher() -> tuple[object, object]:
    global _FETCHER_CACHE
    if _FETCHER_CACHE is not None:
        return _FETCHER_CACHE
    try:
        module = importlib.import_module(_FETCHER_MODULE)
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise TranscriptServiceError(f"Transcript module unavailable: {exc}") from exc
    fetcher_factory = module.create_transcript_fetcher
    formatter = module.format_as_txt
    _FETCHER_CACHE = (fetcher_factory, formatter)
    return _FETCHER_CACHE


def _coerce_segments(raw_segments: list[dict]) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for item in raw_segments:
        try:
            start = float(item.get("start", 0.0))
            duration = float(item.get("duration", 0.0))
            text = str(item.get("text", "")).strip()
        except (TypeError, ValueError):
            continue
        if not text:
            continue
        segments.append({"start": start, "duration": duration, "text": text})
    return segments


def fetch_transcript(
    video_id: str,
    *,
    languages: Sequence[str] | None = None,
    translate_to: str = "en",
    timeout: float = 300.0,
    delay: float = 5.0,
    random_delay: bool = True,
    verbose: bool = False,
) -> TranscriptEntry:
    """Return transcript segments and formatted text for ``video_id``."""

    language_prefs = list(languages or ("en", "en-US", "en-GB"))

    log_level = logging.INFO if verbose else logging.WARNING
    logger = logging.getLogger("fpl_influencer_hivemind.transcripts")
    logger.setLevel(log_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setLevel(log_level)

    try:
        factory, formatter = _load_fetcher()
        fetcher = factory(
            rapidapi_key=os.environ.get("RAPIDAPI_EASYSUB_API_KEY"),
            cookies_path=os.environ.get("YOUTUBE_COOKIES_PATH"),
            max_retries=3,
            retry_backoff=1.5,
            timeout=timeout,
            delay=delay,
            random_delay=random_delay,
            api_method="auto",
        )

        transcript_data, _language, _translated = fetcher.fetch_transcript(
            video_id, language_prefs, translate_to
        )
    except Exception as exc:  # pragma: no cover - relies on networked services
        raise TranscriptServiceError(f"Failed to fetch transcript for {video_id}: {exc}") from exc

    segments = _coerce_segments(transcript_data)
    formatted_text = formatter(transcript_data, include_timestamps=False)

    return {
        "video_id": video_id,
        "text": formatted_text,
        "language": _language,
        "translated": _translated,
        "segments": segments,
    }


def fetch_transcript_text(
    video_id: str,
    *,
    languages: Sequence[str] | None = None,
    translate_to: str = "en",
    timeout: float = 300.0,
    delay: float = 5.0,
    random_delay: bool = True,
    include_timestamps: bool = False,
    verbose: bool = False,
) -> str:
    """Backwards compatible helper returning only the formatted transcript text."""

    transcript = fetch_transcript(
        video_id,
        languages=languages,
        translate_to=translate_to,
        timeout=timeout,
        delay=delay,
        random_delay=random_delay,
        verbose=verbose,
    )

    if include_timestamps:
        factory, formatter = _load_fetcher()
        _ = factory  # pragma: no cover - formatter import side effect for consistency
        segments = [
            {"start": segment["start"], "duration": segment["duration"], "text": segment["text"]}
            for segment in transcript["segments"]
        ]
        return formatter(segments, include_timestamps=True)

    return transcript["text"]


__all__ = ["TranscriptServiceError", "fetch_transcript", "fetch_transcript_text"]
