"""Unit tests for the internal FPL helper modules."""

from __future__ import annotations

import importlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from fpl_influencer_hivemind.fpl import get_current_gameweek as gw
from fpl_influencer_hivemind.fpl import get_my_team as my_team
from fpl_influencer_hivemind.fpl import get_top_ownership as top_ownership
from fpl_influencer_hivemind.fpl import utils


def test_find_current_gameweek_prefers_api_flag() -> None:
    reference = datetime(2025, 1, 1, tzinfo=UTC)
    events = [
        {"id": 5, "name": "GW5", "is_current": True},
        {"id": 6, "name": "GW6", "is_current": False},
    ]
    result = gw.find_current_gameweek(events, reference)
    assert result["id"] == 5
    assert result["method"] == "api_is_current_flag"


def test_find_current_gameweek_uses_deadline_when_flag_missing() -> None:
    reference = datetime(2025, 1, 6, tzinfo=UTC)
    events = [
        {"id": 5, "name": "GW5", "deadline_time": "2025-01-05T11:00:00Z"},
        {"id": 6, "name": "GW6", "deadline_time": "2025-01-12T11:00:00Z"},
    ]
    result = gw.find_current_gameweek(events, reference)
    assert result["id"] == 5
    assert result["method"] == "deadline_calculation_current"


def test_format_gameweek_info_adds_deadline() -> None:
    reference = datetime(2025, 1, 1, tzinfo=UTC)
    gameweek = {
        "id": 5,
        "name": "GW5",
        "is_current": True,
        "is_next": False,
        "is_past": False,
        "deadline_time": "2025-01-05T11:00:00Z",
    }
    formatted = gw.format_gameweek_info(gameweek, reference)
    assert formatted["deadline_passed"] is False
    assert "time_until_deadline" in formatted


@pytest.mark.asyncio
async def test_get_current_gameweek_info(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_create_session():
        return object(), SimpleNamespace(close=lambda: None)

    async def fake_get_bootstrap_data(_client):
        return {
            "events": [
                {
                    "id": 5,
                    "name": "GW5",
                    "is_current": True,
                    "deadline_time": "2025-01-05T11:00:00Z",
                }
            ]
        }

    async def fake_safe_close(_session):
        return None

    monkeypatch.setattr(gw, "create_fpl_session", fake_create_session)
    monkeypatch.setattr(gw, "get_bootstrap_data", fake_get_bootstrap_data)
    monkeypatch.setattr(gw, "safe_close_session", fake_safe_close)

    data = await gw.get_current_gameweek_info()
    assert data["id"] == 5


@pytest.mark.asyncio
async def test_get_top_players_by_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_create_session():
        return object(), SimpleNamespace(close=lambda: None)

    async def fake_get_bootstrap_data(_client):
        return {
            "teams": [{"id": 1, "name": "Team"}],
            "elements": [
                {
                    "id": 10,
                    "web_name": "Player",
                    "team": 1,
                    "element_type": 2,
                    "now_cost": 55,
                    "selected_by_percent": "78.5",
                    "total_points": 100,
                    "minutes": 900,
                    "goals_scored": 5,
                    "assists": 3,
                    "clean_sheets": 4,
                    "goals_conceded": 10,
                    "saves": 0,
                    "bonus": 12,
                    "bps": 200,
                    "form": "5.0",
                    "influence": "200",
                    "creativity": "150",
                    "threat": "100",
                    "ict_index": "45",
                    "expected_points": "6.5",
                    "status": "a",
                    "news": "",
                    "chance_of_playing_next_round": 75,
                }
            ],
        }

    async def fake_safe_close(_session):
        return None

    monkeypatch.setattr(top_ownership, "create_fpl_session", fake_create_session)
    monkeypatch.setattr(top_ownership, "get_bootstrap_data", fake_get_bootstrap_data)
    monkeypatch.setattr(top_ownership, "safe_close_session", fake_safe_close)

    players = await top_ownership.get_top_players_by_ownership(limit=5)
    assert players[0]["web_name"] == "Player"
    assert players[0]["team_name"] == "Team"


@pytest.mark.asyncio
async def test_get_my_team_info(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyUser:
        current_event = 5
        last_deadline_value = 100
        last_deadline_bank = 2
        last_deadline_total_transfers = 3
        id = 123
        player_first_name = "Alex"
        player_last_name = "Smith"
        name = "Test FC"
        summary_overall_points = 2000
        summary_overall_rank = 1234
        summary_event_points = 65
        summary_event_rank = 100000
        favourite_team = 1
        started_event = 1

        async def get_picks(self):
            return {
                5: [
                    {
                        "element": 10,
                        "position": 1,
                        "is_captain": True,
                        "is_vice_captain": False,
                        "multiplier": 2,
                        "selling_price": 55,
                    }
                ]
            }

        async def get_gameweek_history(self):
            return []

        async def get_transfers(self):
            return []

    class DummyFPL:
        async def get_user(self, entry_id: int):
            assert entry_id == 123
            return DummyUser()

    async def fake_create_session():
        return DummyFPL(), SimpleNamespace(close=lambda: None)

    async def fake_get_bootstrap_data(_client):
        return {
            "elements": [
                {
                    "id": 10,
                    "web_name": "Player",
                    "team": 1,
                    "element_type": 2,
                    "now_cost": 55,
                    "total_points": 100,
                }
            ],
            "teams": [{"id": 1, "name": "Team"}],
        }

    async def fake_safe_close(_session):
        return None

    monkeypatch.setattr(my_team, "create_fpl_session", fake_create_session)
    monkeypatch.setattr(my_team, "get_bootstrap_data", fake_get_bootstrap_data)
    monkeypatch.setattr(my_team, "safe_close_session", fake_safe_close)

    data = await my_team.get_my_team_info(entry_id=123)
    assert data["summary"]["team_name"] == "Test FC"
    assert data["current_picks"][0]["web_name"] == "Player"


def test_utils_helpers() -> None:
    assert utils.normalize_price(55) == 5.5
    assert utils.normalize_ownership("12.3") == pytest.approx(12.3)
    assert utils.normalize_ownership(None) == 0.0
    parsed = utils.parse_fpl_datetime("2025-01-05T11:00:00Z")
    assert parsed.tzinfo is not None


def test_load_external_fpl_detects_shadowed_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow_origin = Path(utils.__file__).resolve().parents[3] / "fpl" / "__init__.py"

    class DummySpec:
        origin = str(shadow_origin)

    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: DummySpec() if name == "fpl" else None
    )

    import_called = False

    def fake_import(_name: str):  # pragma: no cover - should not be invoked
        nonlocal import_called
        import_called = True
        return object()

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(ImportError):
        utils._load_external_fpl()

    assert import_called is False
