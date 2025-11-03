"""Client for the youtube-transcript.io API.

The API is used as the primary transcript source because it is substantially
faster and more reliable than the yt-dlp/EasySubAPI combo. The fetcher exposes
an interface compatible with the legacy fetchers so the service layer can keep
its current contract while switching providers transparently.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

import requests  # type: ignore[import-untyped]

_logger = logging.getLogger("fpl_influencer_hivemind.transcripts.youtube_transcript_io")

_API_URL = "https://www.youtube-transcript.io/api/transcripts"


class YouTubeTranscriptIOError(RuntimeError):
    """Raised when the youtube-transcript.io API cannot fulfil a request."""


class YouTubeTranscriptIOFetcher:
    """Thin wrapper around the youtube-transcript.io API."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max(1, max_retries)
        self._retry_backoff = retry_backoff
        self._session = session or requests.Session()

    def fetch_transcript(
        self, video_id: str, languages: Sequence[str], _translate_to: str = "en"
    ) -> tuple[list[dict[str, Any]], str, bool]:
        """Return segments, language, translated flag for ``video_id``.

        ``translate_to`` is accepted for API compatibility but the service does
        not currently support translation. The caller can still detect missing
        transcripts and fall back accordingly.
        """

        payload = {"ids": [video_id]}
        headers = {
            "Authorization": f"Basic {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = self._session.post(
                    _API_URL, json=payload, headers=headers, timeout=self._timeout
                )
                response.raise_for_status()
                data = response.json()
                segments, language = self._parse_response(data, languages)
                return segments, language, False
            except (
                requests.RequestException,
                ValueError,
                YouTubeTranscriptIOError,
            ) as exc:
                last_error = exc
                if attempt < self._max_retries - 1:
                    self._sleep(attempt)
                    continue
                break

        message = f"youtube-transcript.io failed for {video_id}: {last_error}"
        raise YouTubeTranscriptIOError(message) from last_error

    def _sleep(self, attempt: int) -> None:
        delay = self._retry_backoff**attempt
        _logger.warning("Retrying youtube-transcript.io request in %.2fs", delay)
        time.sleep(delay)

    def _parse_response(
        self, response: object, languages: Sequence[str]
    ) -> tuple[list[dict[str, Any]], str]:
        if not isinstance(response, list) or not response:
            raise YouTubeTranscriptIOError("Empty or malformed API response")

        video_payload = response[0]
        if not isinstance(video_payload, dict):
            raise YouTubeTranscriptIOError("Unexpected response structure")

        tracks = video_payload.get("tracks")
        if not isinstance(tracks, list) or not tracks:
            raise YouTubeTranscriptIOError("No transcript tracks available")

        track = self._select_track(tracks, languages)
        raw_segments = track.get("transcript")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise YouTubeTranscriptIOError("Transcript track does not contain segments")

        segments = self._normalise_segments(raw_segments)
        if not segments:
            raise YouTubeTranscriptIOError("Transcript contained no usable segments")

        language = self._extract_language(track)
        return segments, language

    def _select_track(
        self, tracks: Iterable[dict[str, Any]], languages: Sequence[str]
    ) -> dict[str, Any]:
        normalised_prefs = [lang.lower() for lang in languages if lang]
        for preferred in normalised_prefs:
            for track in tracks:
                if not isinstance(track, dict):
                    continue
                codes = {
                    str(track.get("language", "")).lower(),
                    str(track.get("languageCode", "")).lower(),
                    str(track.get("language_code", "")).lower(),
                }
                if preferred and preferred in codes:
                    return track

        for track in tracks:
            if isinstance(track, dict):
                return track
        raise YouTubeTranscriptIOError("No usable transcript tracks found")

    def _normalise_segments(
        self, raw_segments: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for item in raw_segments:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            try:
                start_raw = item.get("start") or item.get("offset") or 0.0
                dur_raw = item.get("dur") or item.get("duration") or 0.0
                start = float(start_raw)
                duration = float(dur_raw)
            except (TypeError, ValueError):
                continue
            segments.append({"start": start, "duration": duration, "text": text})
        return segments

    @staticmethod
    def _extract_language(track: dict[str, Any]) -> str:
        language = track.get("language") or track.get("languageCode")
        if isinstance(language, str) and language:
            return language
        return "unknown"


__all__ = [
    "YouTubeTranscriptIOError",
    "YouTubeTranscriptIOFetcher",
]
