"""Basic smoke tests for the CLI entrypoint."""

import pytest

from fpl_influencer_hivemind import main


def test_main_help_exits() -> None:
    """The CLI should expose help without raising unexpected errors."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
