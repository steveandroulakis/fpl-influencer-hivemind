"""Video discovery strategies used by the hivemind pipeline."""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from ..types import ChannelConfig, ChannelDiscovery, VideoResult
from ..youtube import VideoPickerError, select_single_channel


def _ensure_logger(verbose: bool) -> logging.Logger:
    """Return a configured logger for the video picker module."""

    logger = logging.getLogger("fpl_influencer_hivemind.video_picker")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


def _extract_video_id(url: str) -> str | None:
    """Return the 11-character YouTube video ID from a URL or raw ID string."""

    match = re.search(r"v=([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    if "youtu.be/" in url:
        candidate = url.rsplit("/", 1)[-1]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
            return candidate
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    return None


class DiscoveryStrategy:
    """Strategy interface for selecting videos for a given channel."""

    def discover(
        self,
        *,
        channel: ChannelConfig,
        gameweek_id: int,
        days: int,
        max_per_channel: int,
        verbose: bool,
    ) -> ChannelDiscovery:
        raise NotImplementedError


class HeuristicDiscoveryStrategy(DiscoveryStrategy):
    """Discovery strategy backed by the ``select_single_channel`` heuristics."""

    def discover(
        self,
        *,
        channel: ChannelConfig,
        gameweek_id: int,
        days: int,
        max_per_channel: int,
        verbose: bool,
    ) -> ChannelDiscovery:
        channel_name = channel.get("name", "Unknown channel")
        channel_url = channel.get("url")
        if not channel_url:
            return ChannelDiscovery(
                channel=channel,
                result=None,
                error="Channel configuration missing 'url'",
            )

        logger = _ensure_logger(verbose)

        try:
            selection = select_single_channel(
                channel_name=channel_name,
                channel_url=channel_url,
                gameweek=gameweek_id,
                days_back=days,
                max_per_channel=max_per_channel,
                logger=logger,
            )
        except VideoPickerError as exc:
            return ChannelDiscovery(
                channel=channel,
                result=None,
                error=str(exc),
            )
        except Exception as exc:  # pragma: no cover - defensive
            return ChannelDiscovery(
                channel=channel,
                result=None,
                error=f"Video discovery failed: {exc}",
            )

        video = selection.picked
        if video is None:
            return ChannelDiscovery(
                channel=channel,
                result=None,
                error=f"No video selected for {channel_name}",
            )

        video_id = _extract_video_id(video.url) or ""
        payload: VideoResult = {
            "channel_name": channel_name,
            "video_id": video_id,
            "title": video.title,
            "url": video.url,
            "confidence": selection.confidence,
            "published_at": video.published_at.astimezone(UTC).isoformat(),
            "published_at_formatted": video.published_at.astimezone(UTC).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
            "reasoning": selection.reasoning,
            "matched_signals": selection.matched_signals,
            "gameweek": gameweek_id,
            "generated_at": datetime.now(UTC).isoformat(),
        }

        alternatives = [alt.title for alt in selection.alternatives[:3]]

        return ChannelDiscovery(
            channel=channel,
            result=payload,
            error=None,
            alternatives=alternatives,
        )


__all__ = [
    "DiscoveryStrategy",
    "HeuristicDiscoveryStrategy",
]
