"""Unit tests for the youtube-transcript.io fetcher."""

from __future__ import annotations

import pytest

from fpl_influencer_hivemind.transcripts.youtube_transcript_io_fetcher import (
    YouTubeTranscriptIOError,
    YouTubeTranscriptIOFetcher,
)


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class _FakeSession:
    def __init__(self, payloads: list[object]) -> None:
        self._payloads = payloads
        self.calls: list[tuple[str, dict[str, object]]] = []

    def post(
        self,
        url: str,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: float,
    ) -> _FakeResponse:
        _ = headers, timeout
        self.calls.append((url, json))
        if not self._payloads:
            raise AssertionError("No more fake responses")
        return _FakeResponse(self._payloads.pop(0))


def test_fetcher_returns_segments_using_language_preference() -> None:
    payload = [
        {
            "tracks": [
                {
                    "language": "en",
                    "transcript": [{"start": "0.5", "dur": "1.0", "text": "Hello"}],
                },
                {
                    "language": "es",
                    "transcript": [{"start": "1.5", "dur": "2.0", "text": "Hola"}],
                },
            ]
        }
    ]
    session = _FakeSession([payload])
    fetcher = YouTubeTranscriptIOFetcher("key", session=session)

    segments, language, translated = fetcher.fetch_transcript("abc123", ["es", "en"])

    assert language == "es"
    assert translated is False
    assert segments[0]["start"] == pytest.approx(1.5)
    assert segments[0]["text"] == "Hola"
    assert session.calls[0][0].endswith("/api/transcripts")


def test_fetcher_raises_on_empty_response() -> None:
    session = _FakeSession([[]])
    fetcher = YouTubeTranscriptIOFetcher("key", session=session, max_retries=1)

    with pytest.raises(YouTubeTranscriptIOError):
        fetcher.fetch_transcript("abc123", ["en"])
