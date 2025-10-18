"""Unit tests for the YouTube video discovery helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fpl_influencer_hivemind.youtube import video_picker as vp


class DummyRanker:
    """Anthropic stub that returns a deterministic selection."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def rank_videos_by_channel(self, channels_candidates, _gameweek):
        video = next(iter(channels_candidates.values()))[0]
        return {
            next(iter(channels_candidates.keys())): (
                video,
                vp.AnthropicChannelResponse(
                    channel_name=video.channel_name,
                    chosen_index=0,
                    chosen_url=video.url,
                    confidence=0.95,
                    matched_signals=["team selection"],
                    reasoning="Model preference",
                ),
            )
        }


def test_select_single_channel_returns_ranked_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = vp.VideoItem(
        title="GW5 Team Selection",
        url="https://youtu.be/abc123def45",
        published_at=datetime(2025, 1, 1, tzinfo=UTC),
        channel_name="FPL Test",
        description="My team selection video",
    )

    def fake_fetch(_self, _channel_url: str):
        return [video]

    monkeypatch.setattr(vp.FPLVideoCollector, "_fetch_channel_videos", fake_fetch)
    monkeypatch.setattr(vp, "AnthropicRanker", DummyRanker)

    result = vp.select_single_channel(
        channel_name="FPL Test",
        channel_url="https://www.youtube.com/@fpltest",
        gameweek=5,
        days_back=7,
        max_per_channel=3,
    )

    assert result.picked is video
    assert result.confidence == pytest.approx(0.95)
    assert "GW5" in result.reasoning
    assert "team selection" in result.matched_signals


def test_select_single_channel_raises_when_no_videos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch(_self, _channel_url: str):
        return []

    monkeypatch.setattr(vp.FPLVideoCollector, "_fetch_channel_videos", fake_fetch)
    monkeypatch.setattr(vp, "AnthropicRanker", DummyRanker)

    with pytest.raises(vp.VideoPickerError):
        vp.select_single_channel(
            channel_name="FPL Test",
            channel_url="https://www.youtube.com/@fpltest",
            gameweek=5,
            days_back=7,
            max_per_channel=3,
        )
