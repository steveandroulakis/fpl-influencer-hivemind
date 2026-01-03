"""Constants for the FPL Intelligence Analyzer."""

# Current Premier League teams (2025/2026 season)
PL_TEAMS_2025_26 = [
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Burnley",
    "Chelsea",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Leeds United",
    "Liverpool",
    "Manchester City",
    "Manchester United",
    "Newcastle United",
    "Nottingham Forest",
    "Sunderland",
    "Tottenham Hotspur",
    "West Ham United",
    "Wolverhampton Wanderers",
]

# Prompt block for current PL teams context
PL_TEAMS_CONTEXT = """CURRENT PREMIER LEAGUE TEAMS (2025/2026):
Arsenal, Aston Villa, Bournemouth, Brentford, Brighton, Chelsea, Crystal Palace,
Everton, Fulham, Liverpool, Manchester City, Manchester United, Newcastle United,
Nottingham Forest, Tottenham Hotspur, West Ham United, Wolverhampton Wanderers,
Burnley, Leeds United, Sunderland

IMPORTANT:
- ALL 20 teams above are current Premier League teams
- DO NOT use training knowledge about team league status
- Trust the FPL API data for players and their teams - it's authoritative
"""

__all__ = ["PL_TEAMS_2025_26", "PL_TEAMS_CONTEXT"]
