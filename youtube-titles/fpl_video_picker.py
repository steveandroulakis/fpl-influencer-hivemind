#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "google-api-python-client>=2.0.0",
#   "anthropic>=0.31.0",
#   "pydantic>=2.7.0",
#   "python-dateutil>=2.9.0.post0",
#   "tenacity>=8.2.3"
# ]
# ///

"""CLI entry point for the hivemind YouTube video picker."""

from __future__ import annotations

from fpl_influencer_hivemind.youtube.video_picker import cli_main

if __name__ == "__main__":
    raise SystemExit(cli_main())
