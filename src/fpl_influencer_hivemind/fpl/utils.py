"""Shared helpers for interacting with the public FPL API."""

from __future__ import annotations

import importlib
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import aiohttp
from dateutil.parser import parse as parse_datetime  # type: ignore[import-untyped]

POSITION_MAPPING = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
DEFAULT_TIMEZONE = ZoneInfo("America/Los_Angeles")
FPL_TIMEZONE = ZoneInfo("UTC")

FPLClient = Any

_bootstrap_cache: dict[str, Any] | None = None
_FPL_CLASS: type[Any] | None = None


def _load_external_fpl() -> Any:
    """Import the third-party ``fpl`` package, guarding against local shadowing."""

    spec = importlib.util.find_spec("fpl")
    if spec is None or spec.origin is None:
        raise ImportError(
            "Unable to locate the external 'fpl' package. Install it with 'uv add fpl' "
            "or ensure it is available on PYTHONPATH."
        )

    origin_path = Path(spec.origin)
    repo_root = Path(__file__).resolve().parents[3]
    try:
        resolved_origin = origin_path.resolve()
    except OSError:
        resolved_origin = origin_path

    if resolved_origin.is_absolute():
        try:
            relative_to_repo = resolved_origin.relative_to(repo_root)
        except ValueError:
            relative_to_repo = None

        if relative_to_repo is not None:
            top_level = relative_to_repo.parts[0] if relative_to_repo.parts else ""
            if top_level not in {".venv"}:
                raise ImportError(
                    "Detected a shadowed 'fpl' package inside the repository. Please install "
                    "the upstream 'fpl' client from PyPI so the service helpers can use it."
                )

    module = importlib.import_module("fpl")
    if not hasattr(module, "FPL"):
        raise ImportError(
            "Imported 'fpl' module does not expose the expected 'FPL' client"
        )

    return module


def _ensure_fpl_class() -> type[Any]:
    """Return and cache the external ``FPL`` client class."""

    global _FPL_CLASS
    if _FPL_CLASS is None:
        module = _load_external_fpl()
        _FPL_CLASS = module.FPL

    # Type checker cannot narrow global variables, but we've guaranteed it's not None above
    return cast("type[Any]", _FPL_CLASS)


async def create_fpl_session() -> tuple[FPLClient, aiohttp.ClientSession]:
    session = aiohttp.ClientSession()
    fpl_class = _ensure_fpl_class()
    return fpl_class(session), session


def _to_plain(obj: Any, _seen: set[int] | None = None) -> Any:
    if isinstance(obj, str | int | float | bool | type(None)):
        return obj

    if _seen is None:
        _seen = set()

    obj_id = id(obj)
    if obj_id in _seen:
        return f"<circular reference to {type(obj).__name__}>"

    _seen.add(obj_id)

    try:
        if isinstance(obj, dict):
            return {key: _to_plain(value, _seen) for key, value in obj.items()}
        if hasattr(obj, "__dict__"):
            return {key: _to_plain(value, _seen) for key, value in vars(obj).items()}
        if isinstance(obj, list):
            return [_to_plain(item, _seen) for item in obj]
        return obj
    finally:
        _seen.discard(obj_id)


async def get_bootstrap_data(
    fpl: FPLClient, force_refresh: bool = False
) -> dict[str, Any]:
    global _bootstrap_cache
    if _bootstrap_cache is None or force_refresh:
        teams = await fpl.get_teams()
        players = await fpl.get_players()
        gameweeks = await fpl.get_gameweeks()
        _bootstrap_cache = {
            "teams": [_to_plain(team) for team in teams],
            "elements": [_to_plain(player) for player in players],
            "events": [_to_plain(gameweek) for gameweek in gameweeks],
        }
    return _bootstrap_cache


async def safe_close_session(session: aiohttp.ClientSession) -> None:
    await session.close()


def map_position(element_type: int) -> str:
    return POSITION_MAPPING.get(element_type, "UNK")


def normalize_price(now_cost: int) -> float:
    return round(now_cost / 10.0, 1)


def normalize_ownership(selected_by_percent: str | float | None) -> float:
    if isinstance(selected_by_percent, str):
        try:
            return float(selected_by_percent)
        except (TypeError, ValueError):
            return 0.0
    if selected_by_percent is None:
        return 0.0
    return float(selected_by_percent)


def parse_fpl_datetime(value: str) -> datetime:
    parsed = parse_datetime(value)
    if not isinstance(parsed, datetime):
        raise ValueError(f"Expected datetime, got {type(parsed)}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=FPL_TIMEZONE)
    return parsed.astimezone(FPL_TIMEZONE)


def reset_bootstrap_cache() -> None:
    global _bootstrap_cache
    _bootstrap_cache = None


__all__ = [
    "DEFAULT_TIMEZONE",
    "FPL_TIMEZONE",
    "create_fpl_session",
    "get_bootstrap_data",
    "map_position",
    "normalize_ownership",
    "normalize_price",
    "parse_fpl_datetime",
    "reset_bootstrap_cache",
    "safe_close_session",
]
