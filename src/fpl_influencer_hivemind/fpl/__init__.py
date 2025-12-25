"""FPL API integration modules for gameweek, team, and ownership data."""

from .get_current_gameweek import get_current_gameweek_info
from .get_my_team import get_my_team_info
from .get_top_ownership import get_top_players_by_form, get_top_players_by_ownership

# Note: get_transfer_momentum module is importable via
# `from fpl_influencer_hivemind.fpl import get_transfer_momentum`
# but its function must be accessed as get_transfer_momentum.get_transfer_momentum()

__all__ = [
    "get_current_gameweek_info",
    "get_my_team_info",
    "get_top_players_by_form",
    "get_top_players_by_ownership",  # Deprecated alias
]
