"""Unit tests for the YouTube video discovery helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from fpl_influencer_hivemind.youtube import video_picker as vp


def _fake_llm_pick(
    videos: list[vp.VideoItem],  # noqa: ARG001
    gameweek: int | None,
    model: str,  # noqa: ARG001
    temperature: float,  # noqa: ARG001
    logger: Any,  # noqa: ARG001
) -> tuple[int, float, str, list[str]]:
    """Deterministic stub returning first video with high confidence."""
    return (0, 0.95, f"GW{gameweek} team selection video", ["team selection"])


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

    def fake_fetch(_self: Any, _channel_url: str) -> list[vp.VideoItem]:
        return [video]

    monkeypatch.setattr(vp.FPLVideoCollector, "_fetch_channel_videos", fake_fetch)
    monkeypatch.setattr(vp, "_llm_pick_video", _fake_llm_pick)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

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


def test_select_single_channel_heuristic_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When LLM fails, heuristic fallback picks the video."""
    video = vp.VideoItem(
        title="GW5 Team Selection",
        url="https://youtu.be/abc123def45",
        published_at=datetime(2025, 1, 1, tzinfo=UTC),
        channel_name="FPL Test",
        description="My team selection video",
    )

    def fake_fetch(_self: Any, _channel_url: str) -> list[vp.VideoItem]:
        return [video]

    def fake_llm_fail(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise RuntimeError("API unavailable")

    monkeypatch.setattr(vp.FPLVideoCollector, "_fetch_channel_videos", fake_fetch)
    monkeypatch.setattr(vp, "_llm_pick_video", fake_llm_fail)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    result = vp.select_single_channel(
        channel_name="FPL Test",
        channel_url="https://www.youtube.com/@fpltest",
        gameweek=5,
        days_back=7,
        max_per_channel=3,
    )

    assert result.picked is video
    assert result.confidence > 0


def test_select_single_channel_raises_when_no_videos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch(_self: Any, _channel_url: str) -> list[vp.VideoItem]:
        return []

    monkeypatch.setattr(vp.FPLVideoCollector, "_fetch_channel_videos", fake_fetch)

    with pytest.raises(vp.VideoPickerError):
        vp.select_single_channel(
            channel_name="FPL Test",
            channel_url="https://www.youtube.com/@fpltest",
            gameweek=5,
            days_back=7,
            max_per_channel=3,
        )
