# FPL Data Analysis Scripts

Four production-ready Python scripts for Fantasy Premier League data analysis using the official FPL API.

## Scripts

### 1. `get_team_players.py` - List Players by Premier League Club

Lists all players for a specific Premier League club (team ID 1-20) with comprehensive stats.

**Usage:**
```bash
# By team name (fuzzy matching)
python get_team_players.py --team "Arsenal"
python get_team_players.py --team "Man City"

# By team ID  
python get_team_players.py --team-id 1

# Output to file
python get_team_players.py --team "Liverpool" --format json --out liverpool.json
python get_team_players.py --team-id 2 --format csv --out aston_villa.csv

# Custom fields and sorting
python get_team_players.py --team "Chelsea" --fields "web_name,position,price,total_points" --sort total_points --desc
```

**Key Features:**
- Fuzzy team name matching with conflict resolution
- Comprehensive player stats (position, price, points, form, ownership, etc.)
- Multiple output formats: table, CSV, JSON
- Flexible sorting and field selection

### 2. `get_top_ownership.py` - Top Players by Ownership

Lists the most owned FPL players ranked by `selected_by_percent`.

**Usage:**
```bash
# Top 200 players (default)
python get_top_ownership.py

# Custom limits and filters
python get_top_ownership.py --limit 50
python get_top_ownership.py --limit 100 --only-available --min-minutes 1000

# Output formats
python get_top_ownership.py --limit 25 --format csv --out top25.csv
python get_top_ownership.py --limit 100 --format json --out top100.json

# Custom sorting and fields
python get_top_ownership.py --limit 50 --sort "total_points,selected_by_percent" --fields "web_name,position,price,selected_by_percent,total_points"
```

**Key Features:**
- Rank players by ownership percentage
- Filter by availability status and minimum minutes played
- Multi-field sorting with ascending/descending options
- Rich player stats including ICT index, form, and expected points

### 3. `get_current_gameweek.py` - Current Gameweek Information

Determines the current FPL gameweek with deadline information and timezone handling.

**Usage:**
```bash
# Current gameweek (now)
python get_current_gameweek.py

# Check specific dates
python get_current_gameweek.py --date 2025-08-24
python get_current_gameweek.py --datetime "2025-08-24T15:30"

# UTC timezone handling
python get_current_gameweek.py --datetime "2025-08-24T15:30" --utc

# Save to file
python get_current_gameweek.py --out gameweek.json
```

**Key Features:**
- Smart gameweek detection using API flags and deadline calculation
- Timezone conversion (UTC ↔ America/Los_Angeles)
- Time until/since deadline calculations
- Comprehensive gameweek status information

### 4. `get_my_team.py` - Your FPL Team Information

Get comprehensive information about your specific FPL team using your entry ID.

**Usage:**
```bash
# Basic team info
python get_my_team.py --entry-id 1178124

# Full team analysis
python get_my_team.py --entry-id 1178124 --show "summary,picks,history,transfers"

# Export to JSON
python get_my_team.py --entry-id 1178124 --format json --out my_team.json

# Just current squad as CSV
python get_my_team.py --entry-id 1178124 --show picks --format csv --out squad.csv
```

**Key Features:**
- Complete team summary with manager info, points, and rankings
- Current gameweek squad with captain/vice-captain indicators
- Gameweek-by-gameweek performance history
- Transfer history and team value tracking
- Flexible output sections (summary, picks, history, transfers)

## Installation & Setup

### Option 1: Using uv (Recommended)
Scripts include PEP 723 inline metadata for automatic dependency management:

```bash
# Automatically installs dependencies and runs
uv run fpl/get_team_players.py --team "Arsenal"
uv run fpl/get_top_ownership.py --limit 200
uv run fpl/get_current_gameweek.py
```

### Option 2: Traditional Python
Install dependencies manually:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r fpl/requirements.txt

# Run scripts
python fpl/get_team_players.py --team "Arsenal"
python fpl/get_top_ownership.py --limit 200
python fpl/get_current_gameweek.py
```

### Option 3: Project Dependencies
Install shared dependencies from root `pyproject.toml`:

```bash
# From project root
uv sync  # or pip install -e .

# Run scripts
python fpl/get_team_players.py --team "Arsenal"
```

## Common Command Options

All scripts support these common options:

- `--format {table,json,csv}` - Output format (default: table)
- `--out PATH` - Write to file (format auto-detected by extension)
- `--fields FIELD1,FIELD2,...` - Select specific fields for output
- `--sort FIELD1,FIELD2,...` - Sort by specified fields
- `--asc` / `--desc` - Sort direction (default: descending)

## Data Sources

- **FPL API**: Official Fantasy Premier League API endpoints
- **Library**: Uses `fpl` Python library for reliable data access
- **Reference Team ID**: 1178124 (for examples/testing where entry ID is relevant)
- **Premier League Team IDs**: 1-20 (use `--team-id` or team name matching)

## Output Examples

### Table Format (Default)
```
web_name     | position | price | total_points | minutes | form | selected_by_percent | status
Salah        | MID      | 13.0  | 156          | 1890    | 8.2  | 45.8               | a     
Haaland      | FWD      | 15.0  | 223          | 1654    | 12.4 | 62.1               | a     
```

### JSON Format
```json
[
  {
    "id": 253,
    "web_name": "Salah",
    "position": "MID",
    "price": 13.0,
    "total_points": 156,
    "selected_by_percent": 45.8,
    "team_name": "Liverpool"
  }
]
```

### CSV Format
Standard CSV with headers, suitable for Excel or data analysis tools.

## Error Handling

- **Network Issues**: Graceful handling with clear error messages
- **Invalid Team Names**: Shows fuzzy matches and suggestions
- **API Changes**: Defensive programming for missing/changed fields
- **Timezone Issues**: Robust datetime parsing and conversion

## Future Integration

These scripts are designed for eventual migration to `src/fpl_influencer_hivemind/fpl_tools/` as part of the main package. The utilities in `utils.py` maintain stable interfaces to support this transition without breaking changes.

Key migration targets:
- `utils.py` → `src/fpl_influencer_hivemind/fpl_tools/api_utils.py`
- Script logic → `src/fpl_influencer_hivemind/fpl_tools/` modules  
- CLI interfaces → `src/fpl_influencer_hivemind/cli/` unified command structure

## Dependencies

- **fpl**: Official FPL Python library for API access
- **aiohttp**: Async HTTP client for session management
- **python-dateutil**: Robust datetime parsing
- **Standard Library**: argparse, asyncio, json, csv, pathlib, zoneinfo

All scripts are designed to minimize external dependencies and use standard library functionality wherever possible.