"""Models for the FPL Intelligence Analyzer."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from src.fpl_influencer_hivemind.types import LineupPlan, TransferPlan


class ChannelAnalysis(BaseModel):
    """Pydantic model for individual channel analysis."""

    channel_name: str
    formation: str | None = None
    team_selection: list[str] = []
    transfers_in: list[str] = []
    transfers_out: list[str] = []
    captain_choice: str = "Not specified"
    vice_captain_choice: str = "Not specified"
    key_issues_discussed: list[dict[str, str]] = []
    watchlist: list[dict[str, str]] = []
    bank_itb: str | None = None
    key_reasoning: list[str] = []
    confidence: float = 0.5
    transcript_length: int = 0


@dataclass(slots=True)
class PlayerLookupEntry:
    """Entry in player lookup table."""

    name: str
    position: str
    team: str
    price: float
    element_id: int | None


@dataclass(slots=True)
class SquadPlayerEntry:
    """Entry for a player in the user's squad."""

    name: str
    position: str
    team: str
    selling_price: float


@dataclass(slots=True)
class DecisionOption:
    """A transfer/lineup decision option for the user."""

    label: str
    transfers: TransferPlan
    lineup: LineupPlan
    rationale: str


# Type aliases for common patterns
ConsensusData = dict[str, Any]
SquadContext = dict[str, Any]

__all__ = [
    "ChannelAnalysis",
    "ConsensusData",
    "DecisionOption",
    "PlayerLookupEntry",
    "SquadContext",
    "SquadPlayerEntry",
]
