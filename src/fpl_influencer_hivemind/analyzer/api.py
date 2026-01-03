"""Anthropic API wrapper with retry logic."""

import logging
import os
import re
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def init_anthropic_client() -> anthropic.Anthropic:
    """Initialize and return an Anthropic client."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required")
    return anthropic.Anthropic(api_key=api_key)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def make_anthropic_call(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    system: str,
    max_tokens: int = 3500,
) -> tuple[str, str]:
    """Make a call to the Anthropic API with retry logic.

    Returns:
        Tuple of (response_text, stop_reason)
    """
    logger.debug(f"Making API call to {model}")

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.1,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )

    parts: list[str] = []
    for block in message.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    response_text = "".join(parts).strip()

    stop_reason = getattr(message, "stop_reason", "") or ""
    usage = getattr(message, "usage", None)
    output_tokens = getattr(usage, "output_tokens", None) if usage else None
    logger.debug(
        "API call successful (stop_reason=%s, output_tokens=%s, response_length=%s)",
        stop_reason,
        output_tokens,
        len(response_text),
    )

    if stop_reason and stop_reason not in {"end_turn", "stop_sequence"}:
        logger.warning(
            "Anthropic message returned stop_reason='%s' (response chars=%s)",
            stop_reason,
            len(response_text),
        )

    return response_text, stop_reason


def extract_last_json(response: str) -> str:
    """Extract the last valid JSON object from a response with multiple blocks.

    Models sometimes output multiple JSON blocks when they realize an error
    and self-correct. This extracts the last valid JSON block.
    """
    # Find all JSON blocks in markdown fences
    json_blocks = re.findall(
        r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response, re.DOTALL
    )
    if json_blocks:
        # Return the last JSON block (most likely the corrected one)
        return json_blocks[-1]

    # Try to find raw JSON (not in fences)
    # Look for last complete JSON object
    cleaned = response.strip()
    if cleaned.startswith("{"):
        # Find matching closing brace
        brace_count = 0
        end_pos = 0
        for i, char in enumerate(cleaned):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
        if end_pos > 0:
            return cleaned[:end_pos]

    return cleaned


class AnthropicClient:
    """Wrapper class for Anthropic API calls with retry logic."""

    def __init__(self) -> None:
        """Initialize the Anthropic client."""
        self.client = init_anthropic_client()
        self.sonnet_model = "claude-opus-4-5-20251101"
        self.opus_model = "claude-opus-4-5-20251101"
        self.haiku_model = "claude-3-5-haiku-20241022"

    def call(
        self,
        model: str,
        prompt: str,
        system: str,
        max_tokens: int = 3500,
    ) -> tuple[str, str]:
        """Make an API call with retry logic."""
        return make_anthropic_call(self.client, model, prompt, system, max_tokens)

    def call_sonnet(
        self, prompt: str, system: str, max_tokens: int = 3500
    ) -> tuple[str, str]:
        """Make an API call using Sonnet model."""
        return self.call(self.sonnet_model, prompt, system, max_tokens)

    def call_opus(
        self, prompt: str, system: str, max_tokens: int = 3500
    ) -> tuple[str, str]:
        """Make an API call using Opus model."""
        return self.call(self.opus_model, prompt, system, max_tokens)

    def call_haiku(
        self, prompt: str, system: str, max_tokens: int = 3500
    ) -> tuple[str, str]:
        """Make an API call using Haiku model."""
        return self.call(self.haiku_model, prompt, system, max_tokens)


# Singleton pattern for debug content saving
_debug_config: dict[str, Any] = {"save_prompts": True, "prompts_dir": None}


def configure_debug(save_prompts: bool, prompts_dir: Any) -> None:
    """Configure debug settings for saving prompts."""
    _debug_config["save_prompts"] = save_prompts
    _debug_config["prompts_dir"] = prompts_dir


def save_debug_content(filename: str, content: str) -> None:
    """Save debug content to file if prompts directory is set."""
    if _debug_config["save_prompts"] and _debug_config["prompts_dir"]:
        debug_path = _debug_config["prompts_dir"] / filename
        debug_path.write_text(content, encoding="utf-8")
        logger.debug(f"Saved debug content to {debug_path}")


__all__ = [
    "AnthropicClient",
    "configure_debug",
    "extract_last_json",
    "init_anthropic_client",
    "make_anthropic_call",
    "save_debug_content",
]
