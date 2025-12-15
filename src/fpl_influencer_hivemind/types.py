"""Shared type definitions for the hivemind pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NotRequired, TypedDict

from pydantic import BaseModel

# =============================================================================
# Analyzer Stage Models (Pydantic)
# =============================================================================


class PlayerRef(BaseModel):
    """Player reference with position for validation."""

    name: str
    position: str  # GKP/DEF/MID/FWD
    team: str | None = None


class RiskFlag(BaseModel):
    """Risk flag for a player."""

    player: str
    risk: str


class GapAnalysis(BaseModel):
    """Stage 1 output: gaps between my squad and influencer consensus."""

    players_to_sell: list[PlayerRef]
    players_missing: list[PlayerRef]
    risk_flags: list[RiskFlag]
    formation_gaps: list[str]
    captain_gap: str | None = None


class Transfer(BaseModel):
    """Single transfer move."""

    out_player: str  # "Mateta (FWD)"
    out_team: str
    in_player: str  # "Isak (FWD)"
    in_team: str
    in_price: float
    selling_price: float
    cost_delta: float
    backers: list[str]


class TransferPlan(BaseModel):
    """Stage 2 output: specific transfer moves."""

    transfers: list[Transfer]
    total_cost: float
    new_itb: float
    fts_used: int
    fts_remaining: int
    hit_cost: int  # 0 if within FTs, else 4 * (transfers - FTs)
    reasoning: str


class LineupPlan(BaseModel):
    """Stage 3 output: XI + bench selection."""

    starting_xi: list[str]  # 11 "Player (POS)" strings
    bench: list[str]  # 4 players in auto-sub priority
    captain: str
    vice_captain: str
    formation: str  # e.g., "3-5-2"
    reasoning: str


class ValidationResult(BaseModel):
    """Stage 4 output: validation status."""

    valid: bool
    errors: list[str]
    warnings: list[str]
    failed_stage: str | None = None  # "gap", "transfer", "lineup"


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
    "GapAnalysis",
    "LineupPlan",
    "MyTeamPayload",
    "PlayerRef",
    "RiskFlag",
    "TranscriptEntry",
    "TranscriptSegment",
    "Transfer",
    "TransferPlan",
    "ValidationResult",
    "VideoResult",
]
