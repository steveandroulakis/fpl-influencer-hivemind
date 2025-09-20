"""Unit tests for the aggregation pipeline helpers."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fpl_influencer_hivemind.pipeline import (
    AggregationError,
    AggregationOutcome,
    ChannelConfig,
    ChannelDiscovery,
    aggregate,
    default_transcript_prompt,
    generate_unique_path,
)
from fpl_influencer_hivemind.youtube.video_picker import (
    ChannelResult,
    VideoItem,
    VideoPickerError,
)


class FakeRunner:
    """Deterministic stand-in for :class:`SubprocessRunner` used in tests."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.invocations: list[list[str]] = []
        self.scripts_called: list[str] = []

    def run(
        self, args: Sequence[str], *, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        assert isinstance(args, Sequence)
        _ = cwd
        self.invocations.append(list(args))
        script_path = next((Path(part) for part in args if part.endswith(".py")), None)
        if script_path is None:
            raise AssertionError(f"Could not locate script path in command: {args}")
        script = script_path.name
        self.scripts_called.append(script)

        if script == "get_current_gameweek.py":
            output_path = Path(args[args.index("--out") + 1])
            payload = {
                "id": 5,
                "name": "Gameweek 5",
                "is_current": True,
                "now_utc": "2025-01-01T00:00:00+00:00",
                "now_local": "2024-12-31T16:00:00-08:00",
            }
            output_path.write_text(json.dumps(payload), encoding="utf-8")
        elif script == "get_top_ownership.py":
            output_path = Path(args[args.index("--out") + 1])
            players = [
                {
                    "web_name": "Player A",
                    "team_name": "Team",
                    "selected_by_percent": 75.0,
                }
            ]
            output_path.write_text(json.dumps(players), encoding="utf-8")
        elif script == "get_my_team.py":
            output_path = Path(args[args.index("--out") + 1])
            payload = {
                "summary": {"team_name": "Test XI", "total_points": 42},
                "current_picks": [],
            }
            output_path.write_text(json.dumps(payload), encoding="utf-8")
        elif script == "fpl_transcript.py":
            output_path = Path(args[args.index("--out") + 1])
            output_path.write_text("Line one\nLine two", encoding="utf-8")
        else:  # pragma: no cover - defensive
            raise AssertionError(f"Unexpected script invocation: {script}")

        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr=""
        )


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
        gameweek = int(kwargs.get("gameweek", 0))
        calls.append(gameweek)
        if gameweek in failure_set:
            raise VideoPickerError(f"No videos for gameweek {gameweek}")

        video = VideoItem(
            title=title,
            url="https://youtu.be/abc123",
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
        "fpl_influencer_hivemind.pipeline.select_single_channel",
        fake_select_single_channel,
    )
    return calls


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
    runner = FakeRunner(tmp_path)
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    select_calls = stub_select_single_channel(monkeypatch)

    outcome: AggregationOutcome = aggregate(
        team_id=1178124,
        runner=runner,
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
    transcript = data["youtube_analysis"]["transcripts"]["FPL Test"]["transcript"]
    assert transcript == "Line one Line two"

    # Ensure command sequencing executed expected scripts and discovery ran
    invoked_scripts = set(runner.scripts_called)
    assert {
        "get_current_gameweek.py",
        "get_top_ownership.py",
        "get_my_team.py",
        "fpl_transcript.py",
    }.issubset(invoked_scripts)
    assert select_calls == [6]


def test_aggregate_declines_transcripts_when_prompt_returns_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = FakeRunner(tmp_path)
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    stub_select_single_channel(monkeypatch)

    outcome = aggregate(
        team_id=1178124,
        runner=runner,
        channels=channels,
        artifacts_dir=tmp_path,
        auto_approve_transcripts=False,
        transcript_delay=0.0,
        prompt=lambda _: False,
    )

    assert outcome.transcripts_retrieved == 0
    transcript_scripts = [
        Path(call[1]).name for call in runner.invocations if len(call) > 1
    ]
    assert "fpl_transcript.py" not in transcript_scripts


def test_aggregate_invalid_team_id_raises() -> None:
    with pytest.raises(AggregationError):
        aggregate(team_id=0)


class FailingRunner(FakeRunner):
    """Runner that simulates a subprocess failure."""

    def run(
        self, args: Sequence[str], *, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        script_path = next((Path(part) for part in args if part.endswith(".py")), None)
        if script_path is None:
            raise AssertionError(f"Could not locate script path in command: {args}")
        script = script_path.name
        if script == "get_current_gameweek.py":
            raise subprocess.CalledProcessError(returncode=1, cmd=args, stderr="boom")
        return FakeRunner.run(self, args, cwd=cwd)


def test_collect_fpl_data_failure_surface(tmp_path: Path) -> None:
    runner = FailingRunner(tmp_path)
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    with pytest.raises(AggregationError):
        aggregate(
            team_id=1178124, runner=runner, channels=channels, artifacts_dir=tmp_path
        )


def test_aggregate_handles_missing_discovery_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = FakeRunner(tmp_path)
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    calls = stub_select_single_channel(monkeypatch, fail_gameweeks=[5, 6])

    outcome = aggregate(
        team_id=1178124,
        runner=runner,
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
    runner = FakeRunner(tmp_path)
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    calls = stub_select_single_channel(monkeypatch, fail_gameweeks=[6])

    outcome = aggregate(
        team_id=1178124,
        runner=runner,
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


def test_default_transcript_prompt(monkeypatch) -> None:
    discoveries = [
        ChannelDiscovery(
            channel={"name": "FPL Test", "url": "https://example.com"},
            result={"channel_name": "FPL Test", "video_id": "abc", "title": "Title"},
        )
    ]

    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert default_transcript_prompt(discoveries) is True

    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert default_transcript_prompt(discoveries) is False
