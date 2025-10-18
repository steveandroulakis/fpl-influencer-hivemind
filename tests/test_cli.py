"""CLI level tests with monkeypatched pipeline dependencies."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fpl_influencer_hivemind import cli
from fpl_influencer_hivemind.pipeline import AggregationError, AggregationOutcome
from fpl_influencer_hivemind.types import ChannelDiscovery


def _make_outcome(tmp_path: Path) -> AggregationOutcome:
    artifact = Path(tmp_path) / "gw01_team1.json"
    payload = {
        "team_id": 1,
        "gameweek": {"current": 1},
        "fpl_data": {"gameweek_info": {}, "top_players": [], "my_team": {}},
        "youtube_analysis": {
            "channels_processed": 1,
            "videos_discovered": 1,
            "transcripts_retrieved": 0,
            "video_results": [],
            "transcripts": {},
        },
        "summary": {},
        "generated_at": "2025-01-01T00:00:00Z",
    }
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    return AggregationOutcome(
        team_id=1,
        gameweek_id=1,
        result_path=artifact,
        video_results=[],
        transcripts={},
        channels_processed=1,
        videos_discovered=1,
        transcripts_retrieved=0,
    )


def test_collect_command_writes_destination(tmp_path: Path, monkeypatch) -> None:
    outcome = _make_outcome(tmp_path)
    monkeypatch.setattr(cli, "aggregate", lambda **_: outcome)
    destination = Path(tmp_path) / "custom.json"

    status = cli.main(
        [
            "collect",
            "--team-id",
            "1",
            "--output",
            str(destination),
        ]
    )

    assert status == 0
    assert destination.exists()
    data = json.loads(destination.read_text(encoding="utf-8"))
    assert data["team_id"] == 1


def test_pipeline_command_skips_analysis_when_declined(
    tmp_path: Path, monkeypatch
) -> None:
    outcome = _make_outcome(tmp_path)
    monkeypatch.setattr(cli, "aggregate", lambda **_: outcome)
    monkeypatch.setattr(cli, "_confirm", lambda _: False)

    status = cli.main(["pipeline", "--team-id", "1"])
    assert status == 0


def test_pipeline_command_auto_runs_analysis(tmp_path: Path, monkeypatch) -> None:
    outcome = _make_outcome(tmp_path)
    monkeypatch.setattr(cli, "aggregate", lambda **_: outcome)
    monkeypatch.setattr(cli, "_run_analyzer", lambda **_: 0)

    status = cli.main(
        ["pipeline", "--team-id", "1", "--auto-run-analysis", "--free-transfers", "5"]
    )
    assert status == 0


def test_collect_handles_aggregation_error(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "aggregate",
        lambda **_: (_ for _ in ()).throw(AggregationError("boom")),
    )
    status = cli.main(["collect", "--team-id", "1"])
    assert status == 1


def test_run_analyzer_handles_failure(monkeypatch, tmp_path: Path) -> None:
    def raise_error(args, *, check, text, cwd=None):  # type: ignore[no-redef]
        _ = (check, text, cwd)
        raise subprocess.CalledProcessError(returncode=1, cmd=args, stderr="fail")

    monkeypatch.setattr(cli.subprocess, "run", raise_error)
    status = cli._run_analyzer(
        input_path=Path(tmp_path) / "in.json",
        output_path=Path(tmp_path) / "out.md",
        free_transfers=1,
        verbose=False,
    )
    assert status == 1


def test_confirm(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    assert cli._confirm("?") is True
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert cli._confirm("?") is False


def test_default_transcript_prompt(monkeypatch) -> None:
    discoveries = [
        ChannelDiscovery(
            channel={"name": "FPL Test", "url": "https://example.com"},
            result={"channel_name": "FPL Test", "video_id": "abc", "title": "Title"},
        )
    ]

    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert cli.default_transcript_prompt(discoveries) is True

    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert cli.default_transcript_prompt(discoveries) is False
