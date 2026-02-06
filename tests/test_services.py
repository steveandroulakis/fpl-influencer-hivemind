"""Tests for the lightweight service wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fpl_influencer_hivemind.services import fpl as fpl_service
from fpl_influencer_hivemind.services import transcripts as transcript_service
from fpl_influencer_hivemind.transcripts.youtube_transcript_io_fetcher import (
    YouTubeTranscriptIOError,
)

if TYPE_CHECKING:
    from pytest import MonkeyPatch

    from fpl_influencer_hivemind.types import TranscriptEntry, TranscriptSegment


def _patch_run(monkeypatch: MonkeyPatch, result: Any) -> None:
    async def fake_call(*_args: Any, **_kwargs: Any) -> Any:
        return result

    def fake_run(coro: Any) -> Any:
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    monkeypatch.setattr(fpl_service, "_call", fake_call)
    monkeypatch.setattr(fpl_service, "_run", fake_run)


def test_fpl_get_current_gameweek_info(monkeypatch: MonkeyPatch) -> None:
    _patch_run(monkeypatch, {"id": 1})
    data = fpl_service.get_current_gameweek_info()
    assert data == {"id": 1}


def test_fpl_get_top_ownership(monkeypatch: MonkeyPatch) -> None:
    _patch_run(monkeypatch, [{"web_name": "Player"}])
    data = fpl_service.get_top_ownership(limit=10)
    assert data == [{"web_name": "Player"}]


def test_fpl_get_my_team(monkeypatch: MonkeyPatch) -> None:
    _patch_run(monkeypatch, {"summary": {"team_name": "Test"}})
    data = fpl_service.get_my_team(entry_id=123)
    assert data["summary"]["team_name"] == "Test"


def test_transcripts_fetch_transcript_text(monkeypatch: MonkeyPatch) -> None:
    class DummyFetcher:
        def fetch_transcript(
            self,
            video_id: str,
            _languages: list[str],
            _translate_to: str,
        ) -> tuple[list[TranscriptSegment], str, bool]:
            assert video_id == "abc123"
            return (
                [
                    {"start": 0.0, "duration": 1.0, "text": "Line one"},
                    {"start": 1.0, "duration": 1.0, "text": "Line two"},
                ],
                "en",
                False,
            )

    monkeypatch.setattr(
        transcript_service,
        "_load_fetcher",
        lambda: (
            lambda **_: DummyFetcher(),
            lambda data, **_: "\n".join(segment["text"] for segment in data),
        ),
    )

    transcript: TranscriptEntry = transcript_service.fetch_transcript(
        "abc123", verbose=False
    )
    assert transcript["language"] == "en"
    assert transcript["text"] == "Line one\nLine two"
    assert transcript["segments"][0]["text"] == "Line one"

    text = transcript_service.fetch_transcript_text("abc123", verbose=False)
    assert text == "Line one\nLine two"



def test_transcripts_falls_back_when_youtube_fails(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("YOUTUBE_TRANSCRIPT_IO_KEY", "secret-key")
    monkeypatch.setattr(transcript_service, "_YOUTUBE_IO_CACHE", None)

    class FailingFetcher:
        def fetch_transcript(
            self, *_args: Any, **_kwargs: Any
        ) -> tuple[list[TranscriptSegment], str, bool]:
            raise YouTubeTranscriptIOError("boom")

    def fake_make_fetcher(*_args: Any, **_kwargs: Any) -> FailingFetcher:
        return FailingFetcher()

    class LegacyFetcher:
        def fetch_transcript(
            self,
            video_id: str,
            _languages: list[str],
            _translate_to: str,
        ) -> tuple[list[TranscriptSegment], str, bool]:
            assert video_id == "abc123"
            return (
                [{"start": 0.0, "duration": 1.0, "text": "Fallback"}],
                "en",
                False,
            )

    def legacy_factory(**_kwargs: Any) -> LegacyFetcher:
        return LegacyFetcher()

    def legacy_formatter(
        data: list[TranscriptSegment], *, include_timestamps: bool
    ) -> str:
        assert include_timestamps is False
        return "\n".join(item["text"] for item in data)

    monkeypatch.setattr(transcript_service, "_make_youtube_fetcher", fake_make_fetcher)
    monkeypatch.setattr(
        transcript_service,
        "_load_fetcher",
        lambda: (legacy_factory, legacy_formatter),
    )

    transcript = transcript_service.fetch_transcript("abc123", verbose=False)
    assert transcript["text"] == "Fallback"
    assert transcript["segments"][0]["text"] == "Fallback"
