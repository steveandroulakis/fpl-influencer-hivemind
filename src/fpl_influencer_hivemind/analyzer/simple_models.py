"""Models used by the deterministic analyzer pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.fpl_influencer_hivemind.types import ChannelExtraction


@dataclass(slots=True)
class ResolvedPlayer:
    """Resolved player reference with optional element_id."""

    element_id: int | None
    name: str
    position: str
    team: str


@dataclass(slots=True)
class ResolvedChannelExtraction:
    """Channel extraction with resolved player references."""

    channel: str
    video_id: str
    captain: ResolvedPlayer | None
    vice: ResolvedPlayer | None
    transfers_in: list[ResolvedPlayer]
    transfers_out: list[ResolvedPlayer]
    starting_xi: list[ResolvedPlayer]
    bench: list[ResolvedPlayer]
    watchlist: list[ResolvedPlayer]
    chip_plan: list[str]
    unresolved_names: list[str]
    raw: ChannelExtraction


@dataclass(slots=True)
class ConsensusPlayer:
    """Consensus player entry with backers."""

    element_id: int
    name: str
    position: str
    team: str
    backers: list[str]


@dataclass(slots=True)
class ConsensusSummary:
    """Aggregated deterministic consensus counts."""

    total_channels: int
    captains: dict[int, ConsensusPlayer]
    transfers_in: dict[int, ConsensusPlayer]
    transfers_out: dict[int, ConsensusPlayer]
    watchlist: dict[int, ConsensusPlayer]
    chips: dict[str, list[str]]
    unresolved: dict[str, list[str]]


@dataclass(slots=True)
class SquadPlayer:
    """Structured squad entry."""

    element_id: int
    name: str
    position: str
    team: str
    price: float
    selling_price: float


__all__ = [
    "ConsensusPlayer",
    "ConsensusSummary",
    "ResolvedChannelExtraction",
    "ResolvedPlayer",
    "SquadPlayer",
]
