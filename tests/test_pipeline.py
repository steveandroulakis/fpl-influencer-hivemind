"""Unit tests for the aggregation pipeline helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from fpl_influencer_hivemind.pipeline import (
    AggregationError,
    AggregationOutcome,
    aggregate,
    generate_unique_path,
)
from fpl_influencer_hivemind.services.discovery import DiscoveryStrategy
from fpl_influencer_hivemind.services.fpl import FPLServiceError
from fpl_influencer_hivemind.services.transcripts import TranscriptServiceError
from fpl_influencer_hivemind.types import ChannelConfig, ChannelDiscovery
from fpl_influencer_hivemind.youtube.video_picker import (
    ChannelResult,
    VideoItem,
    VideoPickerError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def stub_select_single_channel(
    monkeypatch: pytest.MonkeyPatch,
    *,
    channel_name: str = "FPL Test",
    channel_url: str = "https://www.youtube.com/@fpltest",
    title: str = "GW5 Team Selection",
    fail_gameweeks: Sequence[int] | None = None,
) -> list[int]:
    calls: list[int] = []
    failure_set = {int(item) for item in (fail_gameweeks or [])}

    def fake_select_single_channel(**kwargs: object) -> ChannelResult:
        gameweek_raw = kwargs.get("gameweek", 0)
        gameweek = int(gameweek_raw) if isinstance(gameweek_raw, str | int) else 0
        calls.append(gameweek)
        if gameweek in failure_set:
            raise VideoPickerError(f"No videos for gameweek {gameweek}")

        video = VideoItem(
            title=title,
            url="https://youtu.be/abc123def45",
            published_at=datetime(2025, 1, 1, tzinfo=UTC),
            channel_name=channel_name,
            description="",
        )
        return ChannelResult(
            channel_name=channel_name,
            channel_url=channel_url,
            picked=video,
            alternatives=[],
            confidence=0.9,
            reasoning="Test reasoning",
            matched_signals=["team selection"],
            heuristic_score=5.0,
        )

    monkeypatch.setattr(
        "fpl_influencer_hivemind.services.discovery.select_single_channel",
        fake_select_single_channel,
    )
    return calls


def setup_default_services(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], list[str]]:
    fpl_calls: list[str] = []

    def fake_current_gameweek_info() -> dict[str, object]:
        fpl_calls.append("current_gameweek")
        return {
            "id": 5,
            "name": "Gameweek 5",
            "is_current": True,
            "now_utc": "2025-01-01T00:00:00+00:00",
            "now_local": "2024-12-31T16:00:00-08:00",
        }

    def fake_top_ownership(limit: int = 150) -> list[dict[str, object]]:
        fpl_calls.append(f"top_ownership_{limit}")
        return [
            {
                "web_name": "Player A",
                "team_name": "Team",
                "selected_by_percent": 75.0,
            }
        ]

    def fake_my_team(entry_id: int) -> dict[str, object]:
        fpl_calls.append(f"my_team_{entry_id}")
        return {
            "summary": {"team_name": "Test XI", "total_points": 42, "current_event": 5},
            "current_picks": [],
            "team_value": {"team_value": 100.0, "bank_balance": 1.0},
        }

    monkeypatch.setattr(
        "fpl_influencer_hivemind.pipeline.fpl_service.get_current_gameweek_info",
        fake_current_gameweek_info,
    )
    monkeypatch.setattr(
        "fpl_influencer_hivemind.pipeline.fpl_service.get_top_ownership",
        fake_top_ownership,
    )
    monkeypatch.setattr(
        "fpl_influencer_hivemind.pipeline.fpl_service.get_my_team",
        fake_my_team,
    )

    transcript_calls: list[str] = []

    def fake_fetch_transcript(video_id: str, **_: object) -> dict[str, object]:
        transcript_calls.append(video_id)
        return {
            "video_id": video_id,
            "text": "Line one\nLine two",
            "language": "en",
            "translated": False,
            "segments": [
                {"start": 0.0, "duration": 1.0, "text": "Line one"},
                {"start": 1.0, "duration": 1.0, "text": "Line two"},
            ],
        }

    monkeypatch.setattr(
        "fpl_influencer_hivemind.pipeline.transcript_service.fetch_transcript",
        fake_fetch_transcript,
    )

    return fpl_calls, transcript_calls


def test_generate_unique_path(tmp_path: Path) -> None:
    path = tmp_path / "example.json"
    path.write_text("{}", encoding="utf-8")
    first = generate_unique_path(path)
    first.write_text("{}", encoding="utf-8")
    second = generate_unique_path(path)
    assert first.name == "example-1.json"
    assert second.name == "example-2.json"


def test_aggregate_creates_expected_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    select_calls = stub_select_single_channel(monkeypatch)
    fpl_calls, transcript_calls = setup_default_services(monkeypatch)

    outcome: AggregationOutcome = aggregate(
        team_id=1178124,
        channels=channels,
        artifacts_dir=tmp_path,
        auto_approve_transcripts=True,
        transcript_delay=0.0,
        verbose=False,
    )

    assert outcome.team_id == 1178124
    assert outcome.gameweek_id == 6
    assert outcome.result_path.exists()

    data = json.loads(outcome.result_path.read_text(encoding="utf-8"))
    assert data["team_id"] == 1178124
    assert data["gameweek"]["current"] == 6
    assert data["gameweek"]["source"] == 5
    assert data["gameweek"]["requested"] == 6
    assert data["gameweek"]["fallback_used"] is False
    assert data["youtube_analysis"]["videos_discovered"] == 1
    assert data["youtube_analysis"]["transcripts_retrieved"] == 1
    transcript_payload = data["youtube_analysis"]["transcripts"]["abc123def45"]
    assert transcript_payload["text"] == "Line one\nLine two"
    assert transcript_payload["language"] == "en"
    assert transcript_payload["translated"] is False
    assert transcript_payload["segments"][0]["text"] == "Line one"
    assert data["youtube_analysis"]["transcript_errors"] == {}

    assert fpl_calls == [
        "current_gameweek",
        "top_ownership_150",
        "my_team_1178124",
    ]
    assert transcript_calls == ["abc123def45"]
    assert select_calls == [6]


def test_aggregate_declines_transcripts_when_prompt_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    stub_select_single_channel(monkeypatch)
    setup_default_services(monkeypatch)

    def fail_fetch(*_: object, **__: object) -> str:  # pragma: no cover - defensive
        pytest.fail("Transcript fetch should not be invoked when prompt declines")

    monkeypatch.setattr(
        "fpl_influencer_hivemind.pipeline.transcript_service.fetch_transcript",
        fail_fetch,
    )

    outcome = aggregate(
        team_id=1178124,
        channels=channels,
        artifacts_dir=tmp_path,
        auto_approve_transcripts=False,
        transcript_delay=0.0,
        prompt=lambda _: False,
    )

    assert outcome.transcripts_retrieved == 0


def test_aggregate_skips_transcripts_without_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    stub_select_single_channel(monkeypatch)
    setup_default_services(monkeypatch)

    def fail_fetch(*_: object, **__: object) -> str:  # pragma: no cover - defensive
        pytest.fail("Transcript fetch should not be invoked without a prompt")

    monkeypatch.setattr(
        "fpl_influencer_hivemind.pipeline.transcript_service.fetch_transcript",
        fail_fetch,
    )

    outcome = aggregate(
        team_id=1178124,
        channels=channels,
        artifacts_dir=tmp_path,
        auto_approve_transcripts=False,
        transcript_delay=0.0,
        prompt=None,
    )

    assert outcome.transcripts_retrieved == 0


def test_aggregate_records_transcript_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    stub_select_single_channel(monkeypatch)
    setup_default_services(monkeypatch)

    def fail_fetch(*_: object, **__: object) -> dict[str, object]:
        raise TranscriptServiceError("no transcript")

    monkeypatch.setattr(
        "fpl_influencer_hivemind.pipeline.transcript_service.fetch_transcript",
        fail_fetch,
    )

    outcome = aggregate(
        team_id=1178124,
        channels=channels,
        artifacts_dir=tmp_path,
        auto_approve_transcripts=True,
        transcript_delay=0.0,
    )

    assert outcome.transcripts_retrieved == 0
    assert outcome.transcript_errors
    assert "abc123def45" in outcome.transcript_errors

    data = json.loads(outcome.result_path.read_text(encoding="utf-8"))
    assert data["youtube_analysis"]["transcripts"] == {}
    assert data["youtube_analysis"]["transcripts_retrieved"] == 0
    assert data["youtube_analysis"]["transcript_errors"]["abc123def45"]["error"] == (
        "no transcript"
    )
    assert data["summary"]["failed_transcripts"] == 1


def test_aggregate_invalid_team_id_raises() -> None:
    with pytest.raises(AggregationError):
        aggregate(team_id=0)


def test_collect_fpl_data_failure_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    def boom() -> dict[str, object]:
        raise FPLServiceError("boom")

    monkeypatch.setattr(
        "fpl_influencer_hivemind.pipeline.fpl_service.get_current_gameweek_info",
        boom,
    )

    with pytest.raises(AggregationError):
        aggregate(team_id=1178124, channels=channels, artifacts_dir=tmp_path)


def test_aggregate_handles_missing_discovery_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    calls = stub_select_single_channel(monkeypatch, fail_gameweeks=[5, 6])
    setup_default_services(monkeypatch)

    outcome = aggregate(
        team_id=1178124,
        channels=channels,
        artifacts_dir=tmp_path,
        auto_approve_transcripts=True,
        transcript_delay=0.0,
    )

    assert outcome.videos_discovered == 0
    data = json.loads(outcome.result_path.read_text(encoding="utf-8"))
    assert data["gameweek"]["fallback_used"] is True
    assert calls == [6, 5]


def test_aggregate_falls_back_to_current_when_no_requested_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    calls = stub_select_single_channel(monkeypatch, fail_gameweeks=[6])
    setup_default_services(monkeypatch)

    outcome = aggregate(
        team_id=1178124,
        channels=channels,
        artifacts_dir=tmp_path,
        auto_approve_transcripts=True,
        transcript_delay=0.0,
    )

    data = json.loads(outcome.result_path.read_text(encoding="utf-8"))
    assert data["gameweek"]["requested"] == 6
    assert data["gameweek"]["current"] == 5
    assert data["gameweek"]["fallback_used"] is True
    assert calls == [6, 5]


def test_aggregate_emits_log_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    stub_select_single_channel(monkeypatch)
    setup_default_services(monkeypatch)

    messages: list[tuple[str, str]] = []

    aggregate(
        team_id=1178124,
        channels=channels,
        artifacts_dir=tmp_path,
        auto_approve_transcripts=True,
        transcript_delay=0.0,
        verbose=False,
        log=lambda message, level: messages.append((level, message)),
    )

    assert any(
        "Starting FPL Influencer Hivemind pipeline" in msg for _, msg in messages
    )
    assert any("Transcript fetch complete" in msg for _, msg in messages)


class DummyStrategy(DiscoveryStrategy):
    """Deterministic discovery strategy for testing injection."""

    def __init__(self) -> None:
        self.invocations: list[ChannelConfig] = []

    def discover(
        self,
        *,
        channel: ChannelConfig,
        gameweek_id: int,
        days: int,
        max_per_channel: int,
        verbose: bool,
    ) -> ChannelDiscovery:
        _ = (gameweek_id, days, max_per_channel, verbose)
        self.invocations.append(channel)
        return ChannelDiscovery(channel=channel, result=None, error="No match")


def test_aggregate_allows_custom_discovery_strategy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    setup_default_services(monkeypatch)

    strategy = DummyStrategy()

    outcome = aggregate(
        team_id=1178124,
        channels=channels,
        artifacts_dir=tmp_path,
        auto_approve_transcripts=True,
        transcript_delay=0.0,
        discovery_strategy=strategy,
    )

    assert not outcome.video_results
    assert strategy.invocations
    assert all(call == channels[0] for call in strategy.invocations)
