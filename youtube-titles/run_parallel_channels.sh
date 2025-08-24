#!/bin/bash
# FPL Parallel Channel Processor
# Processes multiple FPL YouTube channels in parallel and collates results

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHANNELS_CONFIG="${SCRIPT_DIR}/channels.json"
PICKER_SCRIPT="${SCRIPT_DIR}/fpl_video_picker.py"
TEMP_DIR="${SCRIPT_DIR}/temp_results"

# Default parameters (can be overridden by command line)
GAMEWEEK=""
DAYS=7
MAX_PER_CHANNEL=6
OUTPUT_FILE=""
VERBOSE=false

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Process FPL YouTube channels in parallel to find team selection videos.

OPTIONS:
    --gameweek N        Target gameweek number (required)
    --days N           Only consider videos from last N days (default: 7)
    --max-per-channel N Maximum videos to fetch per channel (default: 6)
    --output FILE      Output file for final JSON results (default: stdout)
    --verbose          Enable verbose logging
    --help             Show this help message

EXAMPLES:
    $0 --gameweek 2 --days 7 --output gw2_results.json
    $0 --gameweek 3 --verbose
    $0 --gameweek 2 --max-per-channel 10 --days 14

This script reads channel configuration from channels.json and processes
each channel in parallel, then combines results into a single output.
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gameweek)
            GAMEWEEK="$2"
            shift 2
            ;;
        --days)
            DAYS="$2"
            shift 2
            ;;
        --max-per-channel)
            MAX_PER_CHANNEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# Validation
if [[ -z "$GAMEWEEK" ]]; then
    echo "Error: --gameweek is required" >&2
    usage >&2
    exit 1
fi

if [[ ! -f "$CHANNELS_CONFIG" ]]; then
    echo "Error: Channels config not found: $CHANNELS_CONFIG" >&2
    exit 1
fi

if [[ ! -x "$PICKER_SCRIPT" ]]; then
    echo "Error: Picker script not executable: $PICKER_SCRIPT" >&2
    exit 1
fi

# Create temp directory
mkdir -p "$TEMP_DIR"
trap 'rm -rf "$TEMP_DIR"' EXIT

# Extract channel names from config (properly handle names with spaces)
CHANNELS=()
while IFS= read -r channel; do
    CHANNELS+=("$channel")
done < <(jq -r '.channels[].name' "$CHANNELS_CONFIG")

if [[ ${#CHANNELS[@]} -eq 0 ]]; then
    echo "Error: No channels found in config file" >&2
    exit 1
fi

echo "🚀 Starting parallel processing of ${#CHANNELS[@]} channels for gameweek $GAMEWEEK"
if [[ "$VERBOSE" == true ]]; then
    echo "Channels: ${CHANNELS[*]}"
    echo "Parameters: days=$DAYS, max-per-channel=$MAX_PER_CHANNEL"
fi

# Launch parallel jobs
PIDS=()
RESULTS_FILES=()

for channel in "${CHANNELS[@]}"; do
    # Create safe filename
    safe_name=$(echo "$channel" | tr ' ' '_' | tr -cd '[:alnum:]_-')
    result_file="$TEMP_DIR/${safe_name}.json"
    log_file="$TEMP_DIR/${safe_name}.log"
    
    RESULTS_FILES+=("$result_file")
    
    # Build command
    cmd=(
        "$PICKER_SCRIPT"
        --single-channel "$channel"
        --gameweek "$GAMEWEEK" 
        --days "$DAYS"
        --max-per-channel "$MAX_PER_CHANNEL"
        --out "$result_file"
    )
    
    if [[ "$VERBOSE" == true ]]; then
        cmd+=(--verbose)
    fi
    
    # Launch in background
    if [[ "$VERBOSE" == true ]]; then
        echo "Launching: ${cmd[*]}"
        "${cmd[@]}" > "$log_file" 2>&1 &
    else
        "${cmd[@]}" > "$log_file" 2>&1 &
    fi
    
    PIDS+=($!)
done

echo "⏳ Waiting for ${#PIDS[@]} parallel jobs to complete..."

# Wait for all jobs and collect exit codes
FAILED_JOBS=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    channel=${CHANNELS[$i]}
    
    if wait "$pid"; then
        if [[ "$VERBOSE" == true ]]; then
            echo "✅ Completed: $channel"
        fi
    else
        echo "❌ Failed: $channel (PID $pid)" >&2
        FAILED_JOBS=$((FAILED_JOBS + 1))
        if [[ "$VERBOSE" == true ]]; then
            safe_name=$(echo "$channel" | tr ' ' '_' | tr -cd '[:alnum:]_-')
            echo "Error log for $channel:" >&2
            cat "$TEMP_DIR/${safe_name}.log" >&2
        fi
    fi
done

if [[ $FAILED_JOBS -gt 0 ]]; then
    echo "Warning: $FAILED_JOBS out of ${#CHANNELS[@]} channels failed" >&2
fi

# Combine results
echo "📋 Combining results..."

# Create final JSON structure
final_json=$(cat << 'EOF'
{
  "channels": [],
  "gameweek": null,
  "generated_at": null,
  "summary": {
    "total_channels": 0,
    "successful_channels": 0,
    "failed_channels": 0
  }
}
EOF
)

successful=0
failed=0

# Process each result file
for result_file in "${RESULTS_FILES[@]}"; do
    if [[ -f "$result_file" && -s "$result_file" ]]; then
        # Validate JSON and add to final result
        if jq empty "$result_file" 2>/dev/null; then
            final_json=$(echo "$final_json" | jq ".channels += [$(cat "$result_file")]")
            successful=$((successful + 1))
        else
            echo "Warning: Invalid JSON in $result_file" >&2
            failed=$((failed + 1))
        fi
    else
        failed=$((failed + 1))
    fi
done

# Update summary and metadata
final_json=$(echo "$final_json" | jq "
    .gameweek = $GAMEWEEK |
    .generated_at = \"$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)\" |
    .summary.total_channels = ${#CHANNELS[@]} |
    .summary.successful_channels = $successful |
    .summary.failed_channels = $failed
")

# Output results
if [[ -n "$OUTPUT_FILE" ]]; then
    echo "$final_json" > "$OUTPUT_FILE"
    echo "📄 Results written to: $OUTPUT_FILE"
else
    echo "$final_json"
fi

echo "✨ Complete! Successfully processed $successful/${#CHANNELS[@]} channels"

if [[ $failed -gt 0 ]]; then
    exit 1
fi