"""Shared type definitions for the hivemind pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NotRequired, TypedDict


class _VideoResultRequired(TypedDict):
    channel_name: str
    video_id: str
    title: str
    url: str
    confidence: float
    published_at: str


class VideoResult(_VideoResultRequired, total=False):
    """Partial schema emitted by discovery helpers."""

    published_at_formatted: NotRequired[str]
    reasoning: NotRequired[str]
    matched_signals: NotRequired[list[str]]
    gameweek: NotRequired[int]
    generated_at: NotRequired[str]


class TranscriptSegment(TypedDict):
    """Single transcript segment preserving timing metadata."""

    start: float
    duration: float
    text: str


class _TranscriptEntryRequired(TypedDict):
    video_id: str
    text: str
    language: str
    translated: bool
    segments: list[TranscriptSegment]


class TranscriptEntry(_TranscriptEntryRequired, total=False):
    """Transcript payload stored in aggregation artifacts."""


class GameweekInfo(TypedDict, total=False):
    """Subset of the gameweek payload returned by ``get_current_gameweek``."""

    id: int
    name: str
    is_current: bool
    is_next: bool
    is_past: bool
    finished: bool
    data_checked: bool
    calculation_method: str
    now_utc: str
    now_local: str
    deadline_time_utc: str | None
    deadline_time_local: str | None
    time_until_deadline: str | None
    time_since_deadline: str | None
    deadline_passed: bool | None


class MyTeamPayload(TypedDict, total=False):
    """Minimal schema for the ``get_my_team`` JSON output."""

    summary: dict[str, object]
    current_picks: list[dict[str, object]]
    team_value: dict[str, object]


class ChannelsFile(TypedDict):
    """Structure of ``youtube-titles/channels.json``."""

    channels: list[ChannelConfig]


class ChannelConfig(TypedDict, total=False):
    """Channel metadata used for discovery."""

    name: str
    url: str
    description: NotRequired[str]


@dataclass(slots=True)
class ChannelDiscovery:
    """Outcome of attempting to identify a video for a channel."""

    channel: ChannelConfig
    result: VideoResult | None
    error: str | None = None
    alternatives: list[str] = field(default_factory=list)


__all__ = [
    "ChannelConfig",
    "ChannelDiscovery",
    "ChannelsFile",
    "GameweekInfo",
    "MyTeamPayload",
    "TranscriptEntry",
    "TranscriptSegment",
    "VideoResult",
]
