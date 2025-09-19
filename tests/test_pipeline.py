"""Unit tests for the aggregation pipeline helpers."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
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


class FakeRunner:
    """Deterministic stand-in for :class:`SubprocessRunner` used in tests."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.invocations: list[list[str]] = []

    def run(
        self, args: Sequence[str], *, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        assert isinstance(args, Sequence)
        _ = cwd
        self.invocations.append(list(args))
        script = Path(args[1]).name

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
        elif script == "fpl_video_picker.py":
            output_path = Path(args[args.index("--out") + 1])
            payload = {
                "channel_name": "FPL Test",
                "video_id": "abc123",
                "title": "GW5 Team Selection",
                "url": "https://youtu.be/abc123",
                "confidence": 0.9,
                "published_at": "2025-01-01T00:00:00Z",
            }
            output_path.write_text(json.dumps(payload), encoding="utf-8")
        elif script == "fpl_transcript.py":
            output_path = Path(args[args.index("--out") + 1])
            output_path.write_text("Line one\nLine two", encoding="utf-8")
        else:  # pragma: no cover - defensive
            raise AssertionError(f"Unexpected script invocation: {script}")

        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


def test_generate_unique_path(tmp_path: Path) -> None:
    path = tmp_path / "example.json"
    path.write_text("{}", encoding="utf-8")
    first = generate_unique_path(path)
    first.write_text("{}", encoding="utf-8")
    second = generate_unique_path(path)
    assert first.name == "example-1.json"
    assert second.name == "example-2.json"


def test_aggregate_creates_expected_payload(tmp_path: Path) -> None:
    runner = FakeRunner(tmp_path)
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

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

    # Ensure command sequencing executed expected scripts
    invoked_scripts = {Path(call[1]).name for call in runner.invocations}
    assert {
        "get_current_gameweek.py",
        "get_top_ownership.py",
        "get_my_team.py",
        "fpl_video_picker.py",
        "fpl_transcript.py",
    }.issubset(invoked_scripts)


def test_aggregate_declines_transcripts_when_prompt_returns_false(tmp_path: Path) -> None:
    runner = FakeRunner(tmp_path)
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

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
        script = Path(args[1]).name
        if script == "get_current_gameweek.py":
            raise subprocess.CalledProcessError(returncode=1, cmd=args, stderr="boom")
        return FakeRunner.run(self, args, cwd=cwd)


def test_collect_fpl_data_failure_surface(tmp_path: Path) -> None:
    runner = FailingRunner(tmp_path)
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    with pytest.raises(AggregationError):
        aggregate(team_id=1178124, runner=runner, channels=channels, artifacts_dir=tmp_path)


class NoDiscoveryRunner(FakeRunner):
    """Runner that skips writing discovery output for coverage."""

    def run(
        self, args: Sequence[str], *, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        script = Path(args[1]).name
        if script == "fpl_video_picker.py":
            # Skip creating output file to simulate failure
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="", stderr=""
            )
        return FakeRunner.run(self, args, cwd=cwd)


def test_aggregate_handles_missing_discovery_output(tmp_path: Path) -> None:
    runner = NoDiscoveryRunner(tmp_path)
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

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


class FallbackRunner(FakeRunner):
    """Runner that forces fallback from requested to source gameweek."""

    def run(
        self, args: Sequence[str], *, cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        script = Path(args[1]).name
        if script == "fpl_video_picker.py":
            gameweek_index = args.index("--gameweek") + 1
            gameweek_value = int(args[gameweek_index])
            if gameweek_value == 6:
                # Simulate no results for requested gameweek by omitting output
                return subprocess.CompletedProcess(
                    args=args, returncode=0, stdout="", stderr=""
                )
        return FakeRunner.run(self, args, cwd=cwd)


def test_aggregate_falls_back_to_current_when_no_requested_matches(tmp_path: Path) -> None:
    runner = FallbackRunner(tmp_path)
    channels: list[ChannelConfig] = [
        {"name": "FPL Test", "url": "https://www.youtube.com/@fpltest"}
    ]

    outcome = aggregate(
        team_id=1178124,
        runner=runner,
        channels=channels,
        artifacts_dir=tmp_path,
        auto_approve_transcripts=True,
        transcript_delay=0.0,
    )

    assert outcome.gameweek_id == 5
    data = json.loads(outcome.result_path.read_text(encoding="utf-8"))
    assert data["gameweek"]["requested"] == 6
    assert data["gameweek"]["current"] == 5
    assert data["gameweek"]["fallback_used"] is True


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
