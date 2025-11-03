"""FPL API integration implemented as importable helpers."""

from __future__ import annotations

import asyncio
from typing import Any

from ..fpl import (
    get_current_gameweek as _current_gameweek_module,
)
from ..fpl import (
    get_my_team as _my_team_module,
)
from ..fpl import (
    get_top_ownership as _top_ownership_module,
)


class FPLServiceError(RuntimeError):
    """Raised when FPL data cannot be retrieved."""


def _run(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return loop.run_until_complete(coro)  # pragma: no cover - defensive


async def _call(func: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return await func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - network failures
        raise FPLServiceError(str(exc)) from exc


def get_current_gameweek_info() -> dict[str, Any]:
    return _run(_call(_current_gameweek_module.get_current_gameweek_info))  # type: ignore[no-any-return]


def get_top_ownership(limit: int = 150) -> list[dict[str, Any]]:
    return _run(_call(_top_ownership_module.get_top_players_by_ownership, limit))  # type: ignore[no-any-return]


def get_my_team(entry_id: int) -> dict[str, Any]:
    return _run(_call(_my_team_module.get_my_team_info, entry_id))  # type: ignore[no-any-return]


__all__ = [
    "FPLServiceError",
    "get_current_gameweek_info",
    "get_my_team",
    "get_top_ownership",
]
