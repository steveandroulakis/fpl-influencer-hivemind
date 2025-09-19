#!/usr/bin/env bash
# Legacy shim maintained while migrating to the Python implementation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/fpl_data_aggregator.py" "$@"
