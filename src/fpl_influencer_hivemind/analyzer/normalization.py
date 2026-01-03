"""Player name normalization utilities."""

import re
import unicodedata
from typing import Any

from src.fpl_influencer_hivemind.analyzer.models import PlayerLookupEntry


def normalize_name(name: str) -> str:
    """Normalize player names for matching (lowercase, no accents/punctuation)."""
    normalized = unicodedata.normalize("NFKD", name)
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]", "", stripped.lower())


def normalize_player_label(label: str) -> str:
    """Normalize a player label, removing any position suffix."""
    name, _ = split_player_label(label)
    return normalize_name(name)


def split_player_label(label: str) -> tuple[str, str | None]:
    """Split 'Player (POS)' into name + position."""
    cleaned = label.strip()
    if not cleaned:
        return "", None
    match = re.match(r"^(.*?)\s*\((GKP|DEF|MID|FWD)\)\s*$", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).strip(), match.group(2).strip().upper()
    return cleaned, None


def coerce_price(value: object) -> float:
    """Coerce price values to float safely."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def select_lookup_candidate(
    candidates: list[PlayerLookupEntry],
    name_hint: str,
    pos_hint: str | None,
) -> PlayerLookupEntry | None:
    """Select best lookup candidate using optional position/name hints."""
    if not candidates:
        return None

    filtered = candidates
    if pos_hint:
        pos_hint = pos_hint.upper()
        pos_filtered = [c for c in filtered if c.position.upper() == pos_hint]
        if pos_filtered:
            filtered = pos_filtered

    exact = [c for c in filtered if c.name.lower() == name_hint.lower()]
    if len(exact) == 1:
        return exact[0]

    if len(filtered) == 1:
        return filtered[0]

    return None


def canonicalize_player_label(
    label: str,
    player_lookup: dict[str, list[PlayerLookupEntry]],
    pos_hint: str | None = None,
) -> str:
    """Canonicalize a player label to 'web_name (POS)' when possible."""
    if not label or label.strip().lower() == "not specified":
        return label

    name, pos = split_player_label(label)
    if not name:
        return label.strip()

    key = normalize_name(name)
    candidates = player_lookup.get(key, [])
    candidate = select_lookup_candidate(candidates, name, pos or pos_hint)
    if not candidate:
        return label.strip()

    return f"{candidate.name} ({candidate.position})"


def canonicalize_channel_analysis(
    analysis_data: dict[str, Any],
    player_lookup: dict[str, list[PlayerLookupEntry]],
) -> dict[str, Any]:
    """Normalize player labels in channel analysis payload."""

    def canonicalize_list(items: list[str]) -> list[str]:
        return [
            canonicalize_player_label(item, player_lookup) for item in items if item
        ]

    if "team_selection" in analysis_data:
        team_selection = analysis_data.get("team_selection", [])
        if isinstance(team_selection, list):
            analysis_data["team_selection"] = canonicalize_list(team_selection)
        else:
            analysis_data["team_selection"] = []

    if "transfers_in" in analysis_data:
        transfers_in = analysis_data.get("transfers_in", [])
        if isinstance(transfers_in, list):
            analysis_data["transfers_in"] = canonicalize_list(transfers_in)
        else:
            analysis_data["transfers_in"] = []

    if "transfers_out" in analysis_data:
        transfers_out = analysis_data.get("transfers_out", [])
        if isinstance(transfers_out, list):
            analysis_data["transfers_out"] = canonicalize_list(transfers_out)
        else:
            analysis_data["transfers_out"] = []

    captain = analysis_data.get("captain_choice")
    if isinstance(captain, str):
        analysis_data["captain_choice"] = canonicalize_player_label(
            captain, player_lookup
        )

    vice = analysis_data.get("vice_captain_choice")
    if isinstance(vice, str):
        analysis_data["vice_captain_choice"] = canonicalize_player_label(
            vice, player_lookup
        )

    watchlist = analysis_data.get("watchlist", [])
    if isinstance(watchlist, list):
        normalized_watchlist: list[dict[str, Any]] = []
        for item in watchlist:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "")
            normalized_watchlist.append(
                {
                    **item,
                    "name": canonicalize_player_label(str(name), player_lookup),
                }
            )
        analysis_data["watchlist"] = normalized_watchlist

    return analysis_data


def build_player_lookup(
    top_players: list[dict[str, Any]],
    my_team_data: dict[str, Any],
    transfer_momentum: dict[str, Any] | None = None,
) -> dict[str, list[PlayerLookupEntry]]:
    """Build lookup table for canonical player names."""
    lookup: dict[str, list[PlayerLookupEntry]] = {}

    def add_entry(entry: PlayerLookupEntry, raw_key: str) -> None:
        if not raw_key:
            return
        key = normalize_name(raw_key)
        existing = lookup.get(key, [])
        for candidate in existing:
            if (
                candidate.name == entry.name
                and candidate.position == entry.position
                and candidate.team == entry.team
            ):
                return
        existing.append(entry)
        lookup[key] = existing

    def add_player(
        name: str,
        position: str,
        team: str,
        price: float,
        element_id: int | None,
        aliases: list[str] | None = None,
    ) -> None:
        if not name or not position:
            return
        entry = PlayerLookupEntry(
            name=name,
            position=position,
            team=team,
            price=price,
            element_id=element_id,
        )
        add_entry(entry, name)
        if aliases:
            for alias in aliases:
                add_entry(entry, alias)

    for player in top_players:
        aliases = []
        full_name = player.get("full_name")
        if full_name:
            aliases.append(str(full_name))
        first = player.get("first_name")
        second = player.get("second_name")
        if first and second:
            aliases.append(f"{first} {second}")
        add_player(
            player.get("web_name", ""),
            player.get("position", ""),
            player.get("team_name", ""),
            coerce_price(player.get("price", 0.0)),
            player.get("id"),
            aliases=aliases,
        )

    for pick in my_team_data.get("current_picks", []):
        add_player(
            pick.get("web_name", ""),
            pick.get("player_position", ""),
            pick.get("team_name", ""),
            coerce_price(pick.get("price", 0.0)),
            pick.get("element_id"),
        )

    if transfer_momentum:
        for key in ("top_transfers_in", "top_transfers_out", "top_net_transfers"):
            for player in transfer_momentum.get(key, []) or []:
                add_player(
                    player.get("web_name", ""),
                    player.get("position", ""),
                    player.get("team_name", ""),
                    coerce_price(player.get("price", 0.0)),
                    None,
                )

    return lookup


__all__ = [
    "build_player_lookup",
    "canonicalize_channel_analysis",
    "canonicalize_player_label",
    "coerce_price",
    "normalize_name",
    "normalize_player_label",
    "select_lookup_candidate",
    "split_player_label",
]
