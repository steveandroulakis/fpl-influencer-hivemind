"""Deterministic, accuracy-first analyzer pipeline."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

from pydantic import ValidationError

from src.fpl_influencer_hivemind.analyzer.api import (
    AnthropicClient,
    configure_debug,
    extract_last_json,
    save_debug_content,
)
from src.fpl_influencer_hivemind.analyzer.models import (
    DecisionOption,
    PlayerLookupEntry,
)
from src.fpl_influencer_hivemind.analyzer.normalization import (
    build_player_lookup,
    normalize_name,
    select_lookup_candidate,
)
from src.fpl_influencer_hivemind.analyzer.report import assemble_simple_report
from src.fpl_influencer_hivemind.analyzer.simple_models import (
    ConsensusPlayer,
    ConsensusSummary,
    ResolvedChannelExtraction,
    ResolvedPlayer,
    SquadPlayer,
)
from src.fpl_influencer_hivemind.analyzer.validation.mechanical import (
    validate_lineup,
    validate_transfers,
)
from src.fpl_influencer_hivemind.types import (
    ChannelExtraction,
    LineupPlan,
    ScoredGapAnalysis,
    ScoredPlayerRef,
    TranscriptEntry,
    Transfer,
    TransferPlan,
    ValidationResult,
)


class SimpleFPLAnalyzer:
    """Simple, deterministic analyzer using LLMs only for extraction."""

    def __init__(self, verbose: bool = False, save_prompts: bool = True) -> None:
        self._setup_logging(verbose)
        self.logger = logging.getLogger(__name__)
        self.save_prompts = save_prompts
        self.prompts_dir: Path | None = None
        self.client = AnthropicClient()

    def _setup_logging(self, verbose: bool) -> None:
        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def load_aggregated_data(self, input_file: str) -> dict[str, Any]:
        """Load and validate aggregated pipeline JSON."""
        with Path(input_file).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Aggregated data must be a JSON object")
        data = raw
        for key in ("fpl_data", "youtube_analysis", "gameweek"):
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        return data

    def _load_prompt(self, filename: str) -> str:
        prompt_path = Path(__file__).resolve().parent / "prompts" / filename
        return prompt_path.read_text(encoding="utf-8")

    def _transcript_text(self, entry: TranscriptEntry) -> str:
        text = entry.get("text", "")
        if text:
            return text
        segments = entry.get("segments", [])
        return "\n".join(segment.get("text", "") for segment in segments)

    def _extract_channel(
        self,
        channel_name: str,
        video_id: str,
        transcript: TranscriptEntry,
    ) -> ChannelExtraction | None:
        transcript_text = self._transcript_text(transcript)
        if not transcript_text.strip():
            self.logger.warning("Transcript empty for %s", channel_name)
            return None

        prompt_template = self._load_prompt("extract_channel.txt")

        def build_prompt(concise: bool) -> str:
            extra = ""
            if concise:
                extra = (
                    "\n\nCONCISE MODE:\n"
                    "The previous response was too long or invalid. "
                    "Return a shorter JSON response that still matches the schema. "
                    "Prefer fewer items and shorter quotes."
                )
            return (
                f"{prompt_template}{extra}\n\n"
                f"CHANNEL: {channel_name}\n"
                f"VIDEO_ID: {video_id}\n"
                f"TRANSCRIPT:\n{transcript_text}"
            )

        def parse_response(raw: str) -> ChannelExtraction | None:
            try:
                cleaned = extract_last_json(raw)
                parsed = json.loads(cleaned)
                if not isinstance(parsed, dict):
                    raise json.JSONDecodeError("Expected JSON object", cleaned, 0)
                parsed.update({"channel": channel_name, "video_id": video_id})
                return ChannelExtraction(**parsed)
            except (json.JSONDecodeError, ValidationError) as exc:
                self.logger.error("Extraction parse failed for %s: %s", channel_name, exc)
                return None

        prompt = build_prompt(concise=False)
        if self.prompts_dir and self.save_prompts:
            save_debug_content(f"{channel_name}_extract_prompt.txt", prompt)

        response, stop_reason = self.client.call_sonnet(
            prompt=prompt,
            system=(
                "You are extracting structured FPL decisions from transcripts. "
                "Return only valid JSON matching the provided schema."
            ),
            max_tokens=3500,
        )

        if self.prompts_dir and self.save_prompts:
            save_debug_content(f"{channel_name}_extract_response.json", response)

        extraction = parse_response(response)
        if extraction:
            return extraction

        if stop_reason == "max_tokens":
            self.logger.warning(
                "Extraction hit max_tokens for %s; retrying with concise prompt",
                channel_name,
            )
        else:
            self.logger.warning(
                "Retrying extraction for %s with concise prompt", channel_name
            )

        retry_prompt = build_prompt(concise=True)
        if self.prompts_dir and self.save_prompts:
            save_debug_content(f"{channel_name}_extract_prompt_retry.txt", retry_prompt)

        retry_response, _ = self.client.call_sonnet(
            prompt=retry_prompt,
            system=(
                "You are extracting structured FPL decisions from transcripts. "
                "Return only valid JSON matching the provided schema."
            ),
            max_tokens=3500,
        )

        if self.prompts_dir and self.save_prompts:
            save_debug_content(
                f"{channel_name}_extract_response_retry.json", retry_response
            )

        extraction = parse_response(retry_response)
        if extraction:
            return extraction

        try:
            return ChannelExtraction(channel=channel_name, video_id=video_id)
        except ValidationError:
            return None

    def _validate_lineup_shape(self, extraction: ChannelExtraction) -> list[str]:
        warnings: list[str] = []

        def dedupe(names: list[str]) -> list[str]:
            seen: set[str] = set()
            unique: list[str] = []
            for name in names:
                key = normalize_name(name)
                if not key or key in seen:
                    continue
                seen.add(key)
                unique.append(name)
            return unique

        extraction.starting_xi_names = dedupe(extraction.starting_xi_names)
        extraction.bench_names = dedupe(extraction.bench_names)

        overlap = {
            normalize_name(name)
            for name in extraction.starting_xi_names
        } & {
            normalize_name(name)
            for name in extraction.bench_names
        }
        if overlap:
            warnings.append("Lineup has duplicate names between XI and bench")
            extraction.starting_xi_names = []
            extraction.bench_names = []

        if extraction.starting_xi_names and len(extraction.starting_xi_names) != 11:
            warnings.append(
                f"Lineup XI has {len(extraction.starting_xi_names)} players (expected 11)"
            )
            extraction.starting_xi_names = []

        if extraction.bench_names and len(extraction.bench_names) != 4:
            warnings.append(
                f"Bench has {len(extraction.bench_names)} players (expected 4)"
            )
            extraction.bench_names = []

        return warnings

    @staticmethod
    def _collect_candidate_players(
        player_lookup: dict[str, list[PlayerLookupEntry]],
    ) -> list[dict[str, str]]:
        candidates: dict[tuple[str, str, str], dict[str, str]] = {}
        for entries in player_lookup.values():
            for entry in entries:
                key = (entry.name, entry.position, entry.team)
                if key in candidates:
                    continue
                candidates[key] = {
                    "name": entry.name,
                    "position": entry.position,
                    "team": entry.team,
                }
        ordered = list(candidates.values())
        ordered.sort(key=lambda item: item["name"])
        return ordered

    @staticmethod
    def _collect_extraction_names(extraction: ChannelExtraction) -> list[str]:
        names: list[str] = []

        def add_list(values: list[str]) -> None:
            for value in values:
                if value and value.strip():
                    names.append(value.strip())

        add_list(extraction.transfers_in_names)
        add_list(extraction.transfers_out_names)
        add_list(extraction.starting_xi_names)
        add_list(extraction.bench_names)
        add_list(extraction.watchlist_names)

        if extraction.captain_name:
            names.append(extraction.captain_name.strip())
        if extraction.vice_name:
            names.append(extraction.vice_name.strip())

        for rationale in extraction.player_rationales:
            if rationale.player_name and rationale.player_name.strip():
                names.append(rationale.player_name.strip())

        return names

    @staticmethod
    def _collect_unresolved_names(
        names: list[str],
        player_lookup: dict[str, list[PlayerLookupEntry]],
    ) -> list[str]:
        unresolved: list[str] = []
        seen: set[str] = set()
        for name in names:
            cleaned = name.strip()
            if not cleaned:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            candidates = player_lookup.get(normalize_name(cleaned), [])
            if not candidates:
                unresolved.append(cleaned)
        return unresolved

    def _correct_names_with_llm(
        self,
        names: list[str],
        candidates: list[dict[str, str]],
    ) -> dict[str, str]:
        if not names:
            return {}

        prompt_template = self._load_prompt("normalize_names.txt")
        prompt = (
            f"{prompt_template}\n\n"
            f"CANDIDATES:\n{json.dumps(candidates, indent=2)}\n\n"
            f"NAMES:\n{json.dumps(names, indent=2)}"
        )

        if self.prompts_dir and self.save_prompts:
            save_debug_content("normalize_names_prompt.txt", prompt)

        response, _ = self.client.call_sonnet(
            prompt=prompt,
            system=(
                "You normalize player names to the provided candidate list. "
                "Return JSON only."
            ),
            max_tokens=800,
        )

        if self.prompts_dir and self.save_prompts:
            save_debug_content("normalize_names_response.json", response)

        try:
            cleaned = extract_last_json(response)
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            self.logger.error("Name normalization JSON parse failed: %s", exc)
            return {}

        if isinstance(payload, dict) and "mappings" in payload:
            mappings = payload.get("mappings")
        else:
            mappings = payload

        if not isinstance(mappings, dict):
            self.logger.error("Name normalization returned non-dict mappings")
            return {}

        candidate_names = {item["name"] for item in candidates if item.get("name")}
        cleaned_mappings: dict[str, str] = {}
        for raw, canonical in mappings.items():
            if not isinstance(raw, str) or raw.strip() == "":
                continue
            if not isinstance(canonical, str):
                continue
            canonical_clean = canonical.strip()
            if not canonical_clean:
                continue
            if canonical_clean not in candidate_names:
                continue
            cleaned_mappings[raw.strip()] = canonical_clean

        return cleaned_mappings

    @staticmethod
    def _replace_names_in_text(text: str, mappings: dict[str, str]) -> str:
        if not text or not mappings:
            return text
        replacements = [
            (raw, canonical)
            for raw, canonical in mappings.items()
            if canonical and raw != canonical
        ]
        replacements.sort(key=lambda item: len(item[0]), reverse=True)
        updated = text
        for raw, canonical in replacements:
            pattern = re.compile(rf"(?<!\\w){re.escape(raw)}(?!\\w)", re.IGNORECASE)
            updated = pattern.sub(canonical, updated)
        return updated

    def _apply_name_corrections(
        self,
        extraction: ChannelExtraction,
        mappings: dict[str, str],
    ) -> None:
        if not mappings:
            return

        def map_name(value: str) -> str:
            return mappings.get(value, value)

        extraction.captain_name = map_name(extraction.captain_name)
        extraction.vice_name = map_name(extraction.vice_name)
        extraction.transfers_in_names = [
            map_name(name) for name in extraction.transfers_in_names
        ]
        extraction.transfers_out_names = [
            map_name(name) for name in extraction.transfers_out_names
        ]
        extraction.starting_xi_names = [
            map_name(name) for name in extraction.starting_xi_names
        ]
        extraction.bench_names = [map_name(name) for name in extraction.bench_names]
        extraction.watchlist_names = [
            map_name(name) for name in extraction.watchlist_names
        ]

        for rationale in extraction.player_rationales:
            rationale.player_name = map_name(rationale.player_name)

        for issue in extraction.key_issues:
            issue.topic = self._replace_names_in_text(issue.topic, mappings)
            issue.quote = self._replace_names_in_text(issue.quote, mappings)

    def _normalize_extraction_names(
        self,
        extraction: ChannelExtraction,
        player_lookup: dict[str, list[PlayerLookupEntry]],
        candidates: list[dict[str, str]],
    ) -> None:
        names = self._collect_extraction_names(extraction)
        unresolved = self._collect_unresolved_names(names, player_lookup)
        if not unresolved:
            return
        mappings = self._correct_names_with_llm(unresolved, candidates)
        if not mappings:
            return
        self._apply_name_corrections(extraction, mappings)

    def _normalize_key_issues_with_llm(
        self,
        extraction: ChannelExtraction,
        candidates: list[dict[str, str]],
    ) -> None:
        if not extraction.key_issues:
            return

        prompt_template = self._load_prompt("normalize_key_issues.txt")
        issues_payload = [
            {"topic": issue.topic, "quote": issue.quote}
            for issue in extraction.key_issues
        ]
        prompt = (
            f"{prompt_template}\n\n"
            f"CANDIDATES:\n{json.dumps(candidates, indent=2)}\n\n"
            f"KEY_ISSUES:\n{json.dumps(issues_payload, indent=2)}"
        )

        if self.prompts_dir and self.save_prompts:
            save_debug_content("normalize_key_issues_prompt.txt", prompt)

        response, _ = self.client.call_sonnet(
            prompt=prompt,
            system=(
                "You normalize player names in key issues to the provided "
                "candidate list. Return JSON only."
            ),
            max_tokens=800,
        )

        if self.prompts_dir and self.save_prompts:
            save_debug_content("normalize_key_issues_response.json", response)

        try:
            cleaned = extract_last_json(response)
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            self.logger.error("Key issue normalization JSON parse failed: %s", exc)
            return

        if isinstance(payload, dict) and "mappings" in payload:
            mappings = payload.get("mappings")
        else:
            mappings = payload

        if not isinstance(mappings, dict):
            self.logger.error("Key issue normalization returned invalid payload")
            return

        candidate_names = {item["name"] for item in candidates if item.get("name")}
        cleaned_mappings: dict[str, str] = {}
        for raw, canonical in mappings.items():
            if not isinstance(raw, str) or raw.strip() == "":
                continue
            if not isinstance(canonical, str):
                continue
            canonical_clean = canonical.strip()
            if not canonical_clean:
                continue
            if canonical_clean not in candidate_names:
                continue
            cleaned_mappings[raw.strip()] = canonical_clean

        if not cleaned_mappings:
            return

        for issue in extraction.key_issues:
            issue.topic = self._replace_names_in_text(issue.topic, cleaned_mappings)
            issue.quote = self._replace_names_in_text(issue.quote, cleaned_mappings)

    @staticmethod
    def _resolve_name(
        name: str,
        player_lookup: dict[str, list[PlayerLookupEntry]],
    ) -> ResolvedPlayer | None:
        cleaned = name.strip()
        if not cleaned:
            return None
        candidates = player_lookup.get(normalize_name(cleaned), [])
        if not candidates:
            return None
        candidate = select_lookup_candidate(candidates, cleaned, None)
        if candidate is None and len(candidates) == 1:
            candidate = candidates[0]
        if candidate is None:
            return None
        return ResolvedPlayer(
            element_id=candidate.element_id,
            name=candidate.name,
            position=candidate.position,
            team=candidate.team,
        )

    def _resolve_extraction(
        self,
        extraction: ChannelExtraction,
        player_lookup: dict[str, list[PlayerLookupEntry]],
    ) -> ResolvedChannelExtraction:
        unresolved: list[str] = []

        def resolve_list(names: Iterable[str]) -> list[ResolvedPlayer]:
            resolved: list[ResolvedPlayer] = []
            for name in names:
                player = self._resolve_name(name, player_lookup)
                if player is None:
                    unresolved.append(name)
                    continue
                resolved.append(player)
            return resolved

        captain = self._resolve_name(extraction.captain_name, player_lookup)
        if extraction.captain_name and captain is None:
            unresolved.append(extraction.captain_name)
        vice = self._resolve_name(extraction.vice_name, player_lookup)
        if extraction.vice_name and vice is None:
            unresolved.append(extraction.vice_name)

        return ResolvedChannelExtraction(
            channel=extraction.channel,
            video_id=extraction.video_id,
            captain=captain,
            vice=vice,
            transfers_in=resolve_list(extraction.transfers_in_names),
            transfers_out=resolve_list(extraction.transfers_out_names),
            starting_xi=resolve_list(extraction.starting_xi_names),
            bench=resolve_list(extraction.bench_names),
            watchlist=resolve_list(extraction.watchlist_names),
            chip_plan=list(extraction.chip_plan),
            unresolved_names=unresolved,
            raw=extraction,
        )

    @staticmethod
    def _record_consensus(
        bucket: dict[int, ConsensusPlayer],
        player: ResolvedPlayer,
        channel: str,
    ) -> None:
        if player.element_id is None:
            return
        entry = bucket.get(player.element_id)
        if entry is None:
            bucket[player.element_id] = ConsensusPlayer(
                element_id=player.element_id,
                name=player.name,
                position=player.position,
                team=player.team,
                backers=[channel],
            )
        else:
            if channel not in entry.backers:
                entry.backers.append(channel)

    def _build_consensus(
        self,
        extractions: list[ResolvedChannelExtraction],
    ) -> ConsensusSummary:
        captains: dict[int, ConsensusPlayer] = {}
        transfers_in: dict[int, ConsensusPlayer] = {}
        transfers_out: dict[int, ConsensusPlayer] = {}
        watchlist: dict[int, ConsensusPlayer] = {}
        chips: dict[str, list[str]] = {}
        unresolved: dict[str, list[str]] = {}

        for extraction in extractions:
            channel = extraction.channel
            if extraction.captain:
                self._record_consensus(captains, extraction.captain, channel)
            for player in extraction.transfers_in:
                self._record_consensus(transfers_in, player, channel)
            for player in extraction.transfers_out:
                self._record_consensus(transfers_out, player, channel)
            for player in extraction.watchlist:
                self._record_consensus(watchlist, player, channel)
            for chip in extraction.chip_plan:
                chips.setdefault(chip, []).append(channel)
            for name in extraction.unresolved_names:
                if not name:
                    continue
                unresolved.setdefault(name, []).append(channel)

        return ConsensusSummary(
            total_channels=len(extractions),
            captains=captains,
            transfers_in=transfers_in,
            transfers_out=transfers_out,
            watchlist=watchlist,
            chips=chips,
            unresolved=unresolved,
        )

    @staticmethod
    def _to_float(value: object) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    def _player_score(
        self,
        element_id: int,
        top_players_by_id: dict[int, dict[str, Any]],
    ) -> float:
        data = top_players_by_id.get(element_id)
        if not data:
            return 0.0
        expected_points = self._to_float(data.get("ep_next", 0.0))
        form = self._to_float(data.get("form", 0.0))
        total_points = self._to_float(data.get("total_points", 0.0))
        base = form + (total_points / 20.0)
        if expected_points > 0:
            return expected_points + base
        return base

    def _build_gap_analysis(
        self,
        consensus: ConsensusSummary,
        squad: list[SquadPlayer],
        top_players_by_id: dict[int, dict[str, Any]],
    ) -> ScoredGapAnalysis:
        squad_ids = {player.element_id for player in squad}
        total_channels = max(consensus.total_channels, 1)

        missing: list[ScoredPlayerRef] = []
        to_sell: list[ScoredPlayerRef] = []

        for entry in consensus.transfers_in.values():
            if entry.element_id in squad_ids:
                continue
            severity = (len(entry.backers) / total_channels) * 10
            factors = [f"{len(entry.backers)}/{total_channels} channels"]
            ownership = self._to_float(
                top_players_by_id.get(entry.element_id, {}).get(
                    "selected_by_percent", 0.0
                )
            )
            if ownership >= 15:
                factors.append(f"{ownership:.1f}% ownership")
                severity += 1.0
            missing.append(
                ScoredPlayerRef(
                    name=entry.name,
                    position=entry.position,
                    team=entry.team,
                    severity=min(severity, 10.0),
                    severity_factors=factors,
                )
            )

        missing.sort(key=lambda p: p.severity, reverse=True)

        for entry in consensus.transfers_out.values():
            if entry.element_id not in squad_ids:
                continue
            severity = (len(entry.backers) / total_channels) * 10
            factors = [f"{len(entry.backers)}/{total_channels} channels"]
            to_sell.append(
                ScoredPlayerRef(
                    name=entry.name,
                    position=entry.position,
                    team=entry.team,
                    severity=min(severity, 10.0),
                    severity_factors=factors,
                )
            )

        to_sell.sort(key=lambda p: p.severity, reverse=True)

        captain_gap: str | None = None
        captain_severity = 0.0
        if consensus.captains:
            top_captain = max(
                consensus.captains.values(),
                key=lambda entry: len(entry.backers),
            )
            if top_captain.element_id not in squad_ids:
                captain_gap = f"{top_captain.name} ({top_captain.position})"
                captain_severity = (
                    len(top_captain.backers) / total_channels
                ) * 10

        total_severity = sum(p.severity for p in missing + to_sell) + captain_severity

        return ScoredGapAnalysis(
            players_missing=missing,
            players_to_sell=to_sell,
            risk_flags=[],
            formation_gaps=[],
            captain_gap=captain_gap,
            captain_severity=captain_severity,
            total_severity=total_severity,
        )

    def _build_squad(self, my_team: dict[str, Any]) -> list[SquadPlayer]:
        squad: list[SquadPlayer] = []
        for pick in my_team.get("current_picks", []):
            element_id = pick.get("element_id")
            if element_id is None:
                continue
            squad.append(
                SquadPlayer(
                    element_id=int(element_id),
                    name=str(pick.get("web_name", "")),
                    position=str(pick.get("player_position", "")),
                    team=str(pick.get("team_name", "")),
                    price=self._to_float(pick.get("price", 0.0)),
                    selling_price=self._to_float(
                        pick.get("selling_price", pick.get("price", 0.0))
                    ),
                )
            )
        return squad

    def _select_transfers(
        self,
        consensus: ConsensusSummary,
        squad: list[SquadPlayer],
        top_players_by_id: dict[int, dict[str, Any]],
        itb: float,
        free_transfers: int,
        max_transfers: int,
    ) -> tuple[TransferPlan, list[SquadPlayer]]:
        squad_by_id = {player.element_id: player for player in squad}
        club_counts: dict[str, int] = {}
        for player in squad:
            club_counts[player.team] = club_counts.get(player.team, 0) + 1

        def in_score(entry: ConsensusPlayer) -> tuple[int, float]:
            return (len(entry.backers), self._player_score(entry.element_id, top_players_by_id))

        candidates_in = [
            entry
            for entry in consensus.transfers_in.values()
            if entry.element_id not in squad_by_id
        ]
        candidates_in.sort(key=in_score, reverse=True)

        out_counts = {entry.element_id: len(entry.backers) for entry in consensus.transfers_out.values()}

        selected_transfers: list[Transfer] = []
        used_out_ids: set[int] = set()
        current_itb = itb
        updated_squad = list(squad)

        for entry in candidates_in:
            if len(selected_transfers) >= max_transfers:
                break
            in_player_data = top_players_by_id.get(entry.element_id)
            if not in_player_data:
                continue
            in_price = self._to_float(in_player_data.get("price", 0.0))

            out_candidates = [
                player
                for player in updated_squad
                if player.position == entry.position and player.element_id not in used_out_ids
            ]
            if not out_candidates:
                continue

            def out_priority(player: SquadPlayer) -> tuple[int, float, float]:
                consensus_out = out_counts.get(player.element_id, 0)
                score = self._player_score(player.element_id, top_players_by_id)
                return (
                    0 if consensus_out > 0 else 1,
                    -float(consensus_out),
                    score,
                )

            out_candidates.sort(key=out_priority)

            picked_out: SquadPlayer | None = None
            for candidate in out_candidates:
                cost_delta = in_price - candidate.selling_price
                if current_itb - cost_delta < 0:
                    continue
                next_counts = dict(club_counts)
                next_counts[candidate.team] = max(0, next_counts.get(candidate.team, 0) - 1)
                next_counts[entry.team] = next_counts.get(entry.team, 0) + 1
                if next_counts[entry.team] > 3:
                    continue
                picked_out = candidate
                club_counts = next_counts
                current_itb -= cost_delta
                used_out_ids.add(candidate.element_id)
                selected_transfers.append(
                    Transfer(
                        out_player=f"{candidate.name} ({candidate.position})",
                        out_team=candidate.team,
                        in_player=f"{entry.name} ({entry.position})",
                        in_team=entry.team,
                        in_price=in_price,
                        selling_price=candidate.selling_price,
                        cost_delta=cost_delta,
                        backers=entry.backers,
                    )
                )
                updated_squad = [
                    player
                    for player in updated_squad
                    if player.element_id != candidate.element_id
                ]
                updated_squad.append(
                    SquadPlayer(
                        element_id=entry.element_id,
                        name=entry.name,
                        position=entry.position,
                        team=entry.team,
                        price=in_price,
                        selling_price=in_price,
                    )
                )
                break

            if picked_out is None:
                continue

        total_cost = sum(t.cost_delta for t in selected_transfers)
        fts_used = len(selected_transfers)
        fts_remaining = max(0, free_transfers - fts_used)
        hit_cost = max(0, fts_used - free_transfers) * 4

        plan = TransferPlan(
            transfers=selected_transfers,
            total_cost=total_cost,
            new_itb=current_itb,
            fts_used=fts_used,
            fts_remaining=fts_remaining,
            hit_cost=hit_cost,
            reasoning=(
                "Transfers selected from top consensus targets within budget and club limits."
                if selected_transfers
                else "No transfers recommended based on consensus targets."
            ),
        )

        return plan, updated_squad

    def _select_lineup(
        self,
        squad: list[SquadPlayer],
        consensus: ConsensusSummary,
        top_players_by_id: dict[int, dict[str, Any]],
    ) -> LineupPlan:
        by_position: dict[str, list[SquadPlayer]] = {"GKP": [], "DEF": [], "MID": [], "FWD": []}
        for player in squad:
            by_position.setdefault(player.position, []).append(player)

        formations = [
            (3, 4, 3),
            (3, 5, 2),
            (4, 3, 3),
            (4, 4, 2),
            (4, 5, 1),
            (5, 3, 2),
            (5, 4, 1),
        ]

        def top_by_score(players: list[SquadPlayer], count: int) -> list[SquadPlayer]:
            ranked = sorted(
                players,
                key=lambda p: self._player_score(p.element_id, top_players_by_id),
                reverse=True,
            )
            return ranked[:count]

        best_xi: list[SquadPlayer] = []
        best_total = -1.0
        best_formation = ""

        for defenders, mids, fwds in formations:
            if len(by_position["GKP"]) < 1:
                continue
            if len(by_position["DEF"]) < defenders:
                continue
            if len(by_position["MID"]) < mids:
                continue
            if len(by_position["FWD"]) < fwds:
                continue
            xi = []
            xi.extend(top_by_score(by_position["GKP"], 1))
            xi.extend(top_by_score(by_position["DEF"], defenders))
            xi.extend(top_by_score(by_position["MID"], mids))
            xi.extend(top_by_score(by_position["FWD"], fwds))
            total = sum(
                self._player_score(p.element_id, top_players_by_id) for p in xi
            )
            if total > best_total:
                best_total = total
                best_xi = xi
                best_formation = f"{defenders}-{mids}-{fwds}"

        if not best_xi:
            best_xi = list(squad[:11])
            best_formation = ""  # fallback

        xi_ids = {p.element_id for p in best_xi}
        bench_candidates = [p for p in squad if p.element_id not in xi_ids]
        bench_gk = [p for p in bench_candidates if p.position == "GKP"]
        bench_outfield = [p for p in bench_candidates if p.position != "GKP"]
        bench_outfield.sort(
            key=lambda p: self._player_score(p.element_id, top_players_by_id),
            reverse=True,
        )
        bench: list[SquadPlayer] = []
        bench.extend(bench_outfield[:3])
        bench.extend(bench_gk[:1])

        def label(player: SquadPlayer) -> str:
            return f"{player.name} ({player.position})"

        captain = self._select_captain(best_xi, consensus, top_players_by_id)
        vice = self._select_vice(best_xi, captain, top_players_by_id)

        return LineupPlan(
            starting_xi=[label(p) for p in best_xi],
            bench=[label(p) for p in bench],
            captain=captain,
            vice_captain=vice,
            formation=best_formation,
            reasoning="Lineup selected to maximize projected points while meeting formation rules.",
        )

    def _select_captain(
        self,
        xi: list[SquadPlayer],
        consensus: ConsensusSummary,
        top_players_by_id: dict[int, dict[str, Any]],
    ) -> str:
        xi_ids = {player.element_id for player in xi}
        if consensus.captains:
            top_captain = max(
                consensus.captains.values(),
                key=lambda entry: len(entry.backers),
            )
            if top_captain.element_id in xi_ids:
                return f"{top_captain.name} ({top_captain.position})"
        best = max(
            xi, key=lambda p: self._player_score(p.element_id, top_players_by_id)
        )
        return f"{best.name} ({best.position})"

    def _select_vice(
        self,
        xi: list[SquadPlayer],
        captain: str,
        top_players_by_id: dict[int, dict[str, Any]],
    ) -> str:
        cap_name = captain.split(" (")[0]
        remaining = [
            p
            for p in xi
            if normalize_name(p.name) != normalize_name(cap_name)
        ]
        if not remaining:
            return captain
        best = max(
            remaining, key=lambda p: self._player_score(p.element_id, top_players_by_id)
        )
        return f"{best.name} ({best.position})"

    def _build_transfer_options(
        self,
        consensus: ConsensusSummary,
        squad: list[SquadPlayer],
        top_players_by_id: dict[int, dict[str, Any]],
        itb: float,
        free_transfers: int,
    ) -> list[DecisionOption]:
        options: list[DecisionOption] = []

        roll_plan = TransferPlan(
            transfers=[],
            total_cost=0.0,
            new_itb=itb,
            fts_used=0,
            fts_remaining=free_transfers,
            hit_cost=0,
            reasoning="Roll transfer to preserve flexibility.",
        )
        roll_lineup = self._select_lineup(squad, consensus, top_players_by_id)
        options.append(
            DecisionOption(
                label="Roll: No transfers",
                transfers=roll_plan,
                lineup=roll_lineup,
                rationale="Keeps flexibility for future gameweeks.",
            )
        )

        max_consensus = free_transfers if free_transfers > 0 else 0
        consensus_plan, consensus_squad = self._select_transfers(
            consensus,
            squad,
            top_players_by_id,
            itb,
            free_transfers,
            max_consensus,
        )
        consensus_lineup = self._select_lineup(
            consensus_squad, consensus, top_players_by_id
        )
        options.append(
            DecisionOption(
                label="Consensus: Use available FTs",
                transfers=consensus_plan,
                lineup=consensus_lineup,
                rationale="Follows top consensus targets within free transfers.",
            )
        )

        if free_transfers > 1:
            conservative_plan, conservative_squad = self._select_transfers(
                consensus,
                squad,
                top_players_by_id,
                itb,
                free_transfers,
                1,
            )
            conservative_lineup = self._select_lineup(
                conservative_squad, consensus, top_players_by_id
            )
            options.append(
                DecisionOption(
                    label="Conservative: Use 1 FT",
                    transfers=conservative_plan,
                    lineup=conservative_lineup,
                    rationale="Keeps flexibility by limiting to one move.",
                )
            )

        return options

    def _quality_audit(
        self,
        plan: TransferPlan,
        lineup: LineupPlan,
        squad: list[SquadPlayer],
        post_squad: list[SquadPlayer],
    ) -> ValidationResult:
        original = [asdict(player) for player in squad]
        updated = [asdict(player) for player in post_squad]
        errors = validate_transfers(plan, original, updated)
        errors.extend(validate_lineup(lineup, updated))
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=[],
            failed_stage="transfer" if errors else None,
        )

    def run_analysis(
        self,
        input_file: str,
        output_file: str | None = None,
        free_transfers: int = 1,
        commentary: str | None = None,
        narrative: bool = False,
    ) -> None:
        """Run the deterministic analysis pipeline."""
        if output_file and self.save_prompts:
            output_path = Path(output_file)
            self.prompts_dir = output_path.parent / f"{output_path.stem}_prompts"
            self.prompts_dir.mkdir(exist_ok=True)
            configure_debug(self.save_prompts, self.prompts_dir)

        data = self.load_aggregated_data(input_file)
        gameweek = data["gameweek"]["current"]
        top_players = data["fpl_data"]["top_players"]
        my_team = data["fpl_data"]["my_team"]
        transfer_momentum = data["fpl_data"].get("transfer_momentum", {})
        video_results = data["youtube_analysis"]["video_results"]
        transcripts = data["youtube_analysis"]["transcripts"]

        player_lookup = build_player_lookup(top_players, my_team, transfer_momentum)
        candidate_players = self._collect_candidate_players(player_lookup)

        extractions: list[ResolvedChannelExtraction] = []
        raw_extractions: list[ChannelExtraction] = []

        for video in video_results:
            channel_name = video.get("channel_name", "Unknown")
            video_id = video.get("video_id")
            if not video_id or video_id not in transcripts:
                self.logger.warning("No transcript for %s", channel_name)
                continue
            extraction = self._extract_channel(
                channel_name,
                video_id,
                transcripts[video_id],
            )
            if extraction is None:
                continue
            self._normalize_extraction_names(
                extraction,
                player_lookup,
                candidate_players,
            )
            self._normalize_key_issues_with_llm(extraction, candidate_players)
            self._validate_lineup_shape(extraction)
            raw_extractions.append(extraction)
            extractions.append(self._resolve_extraction(extraction, player_lookup))

        if not extractions:
            raise ValueError("No successful channel extractions; cannot proceed")

        consensus = self._build_consensus(extractions)
        squad = self._build_squad(my_team)
        top_players_by_id = {
            int(player.get("id", 0)): player for player in top_players if player.get("id")
        }
        for pick in my_team.get("current_picks", []):
            element_id = pick.get("element_id")
            if element_id is None:
                continue
            element_id = int(element_id)
            if element_id in top_players_by_id:
                continue
            top_players_by_id[element_id] = {
                "id": element_id,
                "web_name": pick.get("web_name", ""),
                "position": pick.get("player_position", ""),
                "team_name": pick.get("team_name", ""),
                "price": pick.get("price", 0.0),
                "total_points": pick.get("total_points", 0),
                "form": pick.get("form", 0.0),
                "ep_next": pick.get("ep_next", 0.0),
            }

        gap = self._build_gap_analysis(consensus, squad, top_players_by_id)

        itb = self._to_float(
            my_team.get("team_value", {}).get("bank_balance", 0.0)
        )
        options = self._build_transfer_options(
            consensus,
            squad,
            top_players_by_id,
            itb,
            free_transfers,
        )

        quality: ValidationResult | None = None
        if options:
            primary = options[0]
            post_squad = self._build_squad_from_transfers(squad, primary.transfers)
            quality = self._quality_audit(primary.transfers, primary.lineup, squad, post_squad)

        squad_names = {player.name for player in squad}
        report = assemble_simple_report(
            consensus=consensus,
            gap=gap,
            decision_options=options,
            gameweek=gameweek,
            extractions=raw_extractions,
            quality_audit=quality,
            condensed_players=top_players,
            squad_names=squad_names,
            commentary=commentary,
        )

        if narrative:
            report = self._summarize_report(report)

        if output_file:
            Path(output_file).write_text(report, encoding="utf-8")
        else:
            print(report)

    def _build_squad_from_transfers(
        self, squad: list[SquadPlayer], transfers: TransferPlan
    ) -> list[SquadPlayer]:
        updated = list(squad)
        for transfer in transfers.transfers:
            out_name = transfer.out_player.split(" (")[0]
            out_key = normalize_name(out_name)
            updated = [
                player
                for player in updated
                if normalize_name(player.name) != out_key
            ]
            in_name = transfer.in_player.split(" (")[0]
            in_pos = transfer.in_player.split("(")[1].rstrip(")")
            updated.append(
                SquadPlayer(
                    element_id=0,
                    name=in_name,
                    position=in_pos,
                    team=transfer.in_team,
                    price=transfer.in_price,
                    selling_price=transfer.in_price,
                )
            )
        return updated

    def _summarize_report(self, report: str) -> str:
        prompt_template = self._load_prompt("narrative_report.txt")
        prompt = f"{prompt_template}\n\nCOMPUTED REPORT:\n{report}"
        response, _ = self.client.call_sonnet(
            prompt=prompt,
            system=(
                "Summarize the computed report. Do not add new facts or players. "
                "Return Markdown only."
            ),
            max_tokens=1200,
        )
        return response.strip()


__all__ = ["ConsensusSummary", "SimpleFPLAnalyzer"]
