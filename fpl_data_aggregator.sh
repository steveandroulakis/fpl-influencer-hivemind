#!/bin/bash
# FPL Data Aggregator
# Comprehensive FPL data collection orchestrator combining all analysis tools

set -euo pipefail

# Configuration
# FIX: Corrected quoting/parentheses so quotes close properly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="${SCRIPT_DIR}/temp_aggregator"
CHANNELS_CONFIG="${SCRIPT_DIR}/youtube-titles/channels.json"

# Default parameters
TEAM_ID=""
VERBOSE=false
OUTPUT_FILE=""

# Predeclare arrays used later even when branches are skipped (strict mode safe)
DISCOVERY_PIDS=()
DISCOVERY_FILES=()
DISCOVERED_VIDEOS=()
TRANSCRIPT_PIDS=()
TRANSCRIPT_FILES=()

# Colors for output
declare -r RED='\033[0;31m'
declare -r GREEN='\033[0;32m'
declare -r YELLOW='\033[1;33m'
declare -r BLUE='\033[0;34m'
declare -r NC='\033[0m' # No Color

# Usage function
usage() {
    cat << EOF
Usage: $0 --team-id TEAM_ID [OPTIONS]

Comprehensive FPL data aggregation combining gameweek info, player ownership, 
team analysis, YouTube video discovery, and transcript analysis.

REQUIRED:
    --team-id N         Your FPL team/entry ID - e.g. 1178124

OPTIONS:
    --output-file FILE  Output comprehensive results to JSON file
    --verbose           Enable verbose logging
    --help              Show this help message

EXAMPLES:
    $0 --team-id 1178124
    $0 --team-id 1178124 --output-file comprehensive_analysis.json
    $0 --team-id 1178124 --verbose --output-file results.json

This script orchestrates:
1. Current gameweek detection
2. Top 150 player ownership analysis  
3. Your team analysis
4. Parallel YouTube video discovery across FPL channels
5. Parallel transcript fetching for discovered videos
6. Comprehensive output compilation

Requires ANTHROPIC_API_KEY environment variable for YouTube analysis.
EOF
}

# Logging functions
log_info() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${BLUE}[INFO]${NC} $*"
    fi
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Progress indicator
show_progress() {
    echo -e "${BLUE}[PROGRESS]${NC} $*"
}

# Cleanup function
cleanup() {
    if [[ -d "$TEMP_DIR" ]]; then
        log_info "Cleaning up temporary files..."
        rm -rf "$TEMP_DIR"
    fi
}

# Setup trap for cleanup
trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --team-id)
            TEAM_ID="$2"
            shift 2
            ;;
        --output-file)
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
if [[ -z "$TEAM_ID" ]]; then
    log_error "--team-id is required"
    usage >&2
    exit 1
fi

if ! [[ "$TEAM_ID" =~ ^[0-9]+$ ]] || [[ "$TEAM_ID" -le 0 ]]; then
    log_error "Team ID must be a positive integer"
    exit 1
fi

if [[ ! -f "$CHANNELS_CONFIG" ]]; then
    log_error "Channels config not found: $CHANNELS_CONFIG"
    exit 1
fi

# Tooling check
if ! command -v jq >/dev/null 2>&1; then
    log_error "jq is required but not installed"
    exit 1
fi

# Check for required environment variables
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    log_warn "ANTHROPIC_API_KEY not set - YouTube analysis may fail"
fi

# Check required scripts (must be executable because we exec them directly)
required_scripts=(
    "fpl/get_current_gameweek.py"
    "fpl/get_top_ownership.py"
    "fpl/get_my_team.py" 
    "youtube-titles/fpl_video_picker.py"
    "youtube-transcript/fpl_transcript.py"
)

for script in "${required_scripts[@]}"; do
    script_path="${SCRIPT_DIR}/${script}"
    if [[ ! -x "$script_path" ]]; then
        log_error "Required script not found or not executable: $script_path"
        exit 1
    fi
done

# Create temp directory
mkdir -p "$TEMP_DIR"
log_info "Created temporary directory: $TEMP_DIR"

show_progress "Starting FPL Data Aggregation for team ID: $TEAM_ID"

# Phase 1: FPL Data Collection
show_progress "Phase 1: Collecting FPL Data"

# Get current gameweek
log_info "Fetching current gameweek..."
GAMEWEEK_FILE="${TEMP_DIR}/gameweek.json"
if "${SCRIPT_DIR}/fpl/get_current_gameweek.py" --out "$GAMEWEEK_FILE" > /dev/null 2>&1; then
    GAMEWEEK=$(jq -r '.id' "$GAMEWEEK_FILE")
    log_success "Current gameweek: $GAMEWEEK"
else
    log_error "Failed to get current gameweek"
    exit 1
fi

# Get top 150 players by ownership
log_info "Fetching top 150 players by ownership..."
TOP_PLAYERS_FILE="${TEMP_DIR}/top_players.json"  
if "${SCRIPT_DIR}/fpl/get_top_ownership.py" --limit 150 --format json --out "$TOP_PLAYERS_FILE" > /dev/null 2>&1; then
    PLAYER_COUNT=$(jq 'length' "$TOP_PLAYERS_FILE")
    log_success "Retrieved $PLAYER_COUNT top players"
else
    log_error "Failed to get top players"
    exit 1
fi

# Get my team information
log_info "Fetching team information for ID: $TEAM_ID..."
MY_TEAM_FILE="${TEMP_DIR}/my_team.json"
if "${SCRIPT_DIR}/fpl/get_my_team.py" --entry-id "$TEAM_ID" --format json --out "$MY_TEAM_FILE" > /dev/null 2>&1; then
    TEAM_NAME=$(jq -r '.summary.team_name' "$MY_TEAM_FILE")
    TOTAL_POINTS=$(jq -r '.summary.total_points' "$MY_TEAM_FILE")
    log_success "Retrieved team data: $TEAM_NAME - $TOTAL_POINTS points"
else
    log_error "Failed to get team information for ID: $TEAM_ID"
    exit 1
fi

# Phase 2: YouTube Video Discovery - Parallel
show_progress "Phase 2: YouTube Video Discovery - Parallel Processing"

# Extract channel names from config
CHANNELS=()
while IFS= read -r channel; do
    CHANNELS+=("$channel")
done < <(jq -r '.channels[].name' "$CHANNELS_CONFIG")

log_info "Processing ${#CHANNELS[@]} channels in parallel for gameweek $GAMEWEEK"

# Launch parallel video discovery jobs
DISCOVERY_PIDS=()
DISCOVERY_FILES=()

for channel in "${CHANNELS[@]}"; do
    # Create safe filename
    safe_name=$(echo "$channel" | tr ' ' '_' | tr -cd '[:alnum:]_-')
    discovery_file="${TEMP_DIR}/videos_${safe_name}.json"
    log_file="${TEMP_DIR}/discovery_${safe_name}.log"
    
    DISCOVERY_FILES+=("$discovery_file")
    
    # Build command
    cmd=(
        "${SCRIPT_DIR}/youtube-titles/fpl_video_picker.py"
        --single-channel "$channel"
        --gameweek "$GAMEWEEK"
        --days 7
        --max-per-channel 6
        --out "$discovery_file"
    )
    
    if [[ "$VERBOSE" == true ]]; then
        cmd+=(--verbose)
        log_info "Launching video discovery: ${cmd[*]}"
    fi
    
    # Launch in background
    "${cmd[@]}" > "$log_file" 2>&1 &
    DISCOVERY_PIDS+=($!)
done

log_info "Waiting for ${#DISCOVERY_PIDS[@]} video discovery jobs to complete..."

# Wait for video discovery jobs and collect results
DISCOVERY_FAILED=0
DISCOVERED_VIDEOS=()

for i in "${!DISCOVERY_PIDS[@]}"; do
    pid=${DISCOVERY_PIDS[$i]}
    channel=${CHANNELS[$i]}
    discovery_file=${DISCOVERY_FILES[$i]}
    
    if wait "$pid"; then
        if [[ -f "$discovery_file" && -s "$discovery_file" ]]; then
            # Extract video ID from result
            video_id=$(jq -r '.video_id // empty' "$discovery_file" 2>/dev/null || echo "")
            if [[ -n "$video_id" && "$video_id" != "null" ]]; then
                DISCOVERED_VIDEOS+=("$video_id")
                log_success "Discovered video for $channel: $video_id"
            else
                log_warn "No video found for $channel"
            fi
        else
            log_warn "No output file or empty result for $channel"
        fi
    else
        log_error "Video discovery failed for $channel - PID $pid"
        DISCOVERY_FAILED=$((DISCOVERY_FAILED + 1))
        if [[ "$VERBOSE" == true ]]; then
            safe_name=$(echo "$channel" | tr ' ' '_' | tr -cd '[:alnum:]_-')
            log_file="${TEMP_DIR}/discovery_${safe_name}.log"
            if [[ -f "$log_file" ]]; then
                echo "Error log for $channel:" >&2
                cat "$log_file" >&2
            fi
        fi
    fi
done

log_success "Video discovery complete. Found ${#DISCOVERED_VIDEOS[@]} videos"
if [[ $DISCOVERY_FAILED -gt 0 ]]; then
    log_warn "$DISCOVERY_FAILED out of ${#CHANNELS[@]} channels failed"
fi

# Phase 3: Sequential Transcript Fetching with Rate Limiting
TRANSCRIPT_FAILED=0
SUCCESSFUL_TRANSCRIPTS=0
TRANSCRIPT_DELAY=10  # 10 second delay between requests

if [[ ${#DISCOVERED_VIDEOS[@]} -gt 0 ]]; then
    show_progress "Phase 3: Sequential Transcript Fetching (Rate Limited)"
    
    if [[ ${#DISCOVERED_VIDEOS[@]} -gt 1 ]]; then
        log_info "Processing ${#DISCOVERED_VIDEOS[@]} videos sequentially with ${TRANSCRIPT_DELAY}s delays to avoid rate limiting"
    else
        log_info "Processing ${#DISCOVERED_VIDEOS[@]} video"
    fi
    
    TRANSCRIPT_FILES=()
    video_num=0
    
    for video_id in "${DISCOVERED_VIDEOS[@]}"; do
        video_num=$((video_num + 1))
        transcript_file="${TEMP_DIR}/transcript_${video_id}.txt"
        log_file="${TEMP_DIR}/transcript_${video_id}.log"
        
        TRANSCRIPT_FILES+=("$transcript_file")
        
        # Add delay between requests (skip for first video)
        if [[ $video_num -gt 1 ]]; then
            log_info "Waiting ${TRANSCRIPT_DELAY} seconds before processing next transcript..."
            sleep $TRANSCRIPT_DELAY
        fi
        
        log_info "Processing video $video_num of ${#DISCOVERED_VIDEOS[@]}: $video_id"
        
        # Build command with delay and cookie support
        cmd=(
            "${SCRIPT_DIR}/youtube-transcript/fpl_transcript.py"
            --id "$video_id"
            --format txt
            --out "$transcript_file"
            --delay 5
            --random-delay
        )
        
        if [[ "$VERBOSE" == true ]]; then
            cmd+=(--verbose)
            log_info "Executing: ${cmd[*]}"
        fi
        
        # Execute synchronously (no background process)
        if "${cmd[@]}" > "$log_file" 2>&1; then
            if [[ -f "$transcript_file" && -s "$transcript_file" ]]; then
                SUCCESSFUL_TRANSCRIPTS=$((SUCCESSFUL_TRANSCRIPTS + 1))
                log_success "Retrieved transcript for video: $video_id"
            else
                log_warn "No transcript available for video: $video_id"
            fi
        else
            log_error "Transcript fetch failed for video: $video_id"
            TRANSCRIPT_FAILED=$((TRANSCRIPT_FAILED + 1))
            if [[ "$VERBOSE" == true ]]; then
                if [[ -f "$log_file" ]]; then
                    echo "Error log for $video_id:" >&2
                    cat "$log_file" >&2
                fi
            fi
        fi
    done
    
    log_success "Sequential transcript fetching complete. Retrieved $SUCCESSFUL_TRANSCRIPTS out of ${#DISCOVERED_VIDEOS[@]} transcripts"
    if [[ $TRANSCRIPT_FAILED -gt 0 ]]; then
        log_warn "$TRANSCRIPT_FAILED transcript(s) failed (may be due to IP blocking or rate limiting)"
    fi
else
    log_warn "No videos discovered, skipping transcript fetching"
fi

# Phase 4: Result Compilation & Output
show_progress "Phase 4: Compiling Results"

# Compute success rate safely (avoid division by zero)
if (( ${#CHANNELS[@]} > 0 )); then
    SUCCESS_RATE=$(( ${#DISCOVERED_VIDEOS[@]} * 100 / ${#CHANNELS[@]} ))
else
    SUCCESS_RATE=0
fi

# Create comprehensive results
RESULTS_FILE="${TEMP_DIR}/comprehensive_results.json"

# Build JSON structure
cat > "$RESULTS_FILE" << EOF
{
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)",
  "team_id": $TEAM_ID,
  "gameweek": {
    "current": $GAMEWEEK
  },
  "fpl_data": {
    "gameweek_info": $(cat "$GAMEWEEK_FILE"),
    "top_players": $(cat "$TOP_PLAYERS_FILE"),
    "my_team": $(cat "$MY_TEAM_FILE")
  },
  "youtube_analysis": {
    "channels_processed": ${#CHANNELS[@]},
    "videos_discovered": ${#DISCOVERED_VIDEOS[@]},
    "transcripts_retrieved": $SUCCESSFUL_TRANSCRIPTS,
    "video_results": [],
    "transcripts": {}
  },
  "summary": {
    "total_channels": ${#CHANNELS[@]},
    "failed_discoveries": $DISCOVERY_FAILED,
    "failed_transcripts": $TRANSCRIPT_FAILED,
    "success_rate": "${SUCCESS_RATE}%"
  }
}
EOF

# Add video discovery results
for discovery_file in "${DISCOVERY_FILES[@]}"; do
    if [[ -f "$discovery_file" && -s "$discovery_file" ]]; then
        # Add to video_results array (slurpfile gives [obj], += appends element)
        jq --slurpfile new_result "$discovery_file" \
           '.youtube_analysis.video_results += $new_result' \
           "$RESULTS_FILE" > "${RESULTS_FILE}.tmp" && mv "${RESULTS_FILE}.tmp" "$RESULTS_FILE"
    fi
done

# Add transcript data  
for transcript_file in "${TRANSCRIPT_FILES[@]}"; do
    if [[ -f "$transcript_file" && -s "$transcript_file" ]]; then
        # Extract video ID from filename
        video_id=$(basename "$transcript_file" .txt | sed 's/transcript_//')
        
        # Clean transcript: remove newlines, extra spaces, and create continuous text
        transcript_content=$(cat "$transcript_file" | tr '\n' ' ' | tr -s ' ' | sed 's/^ *//;s/ *$//' | jq -Rs .)
        jq --arg video_id "$video_id" --argjson transcript "$transcript_content" \
           '.youtube_analysis.transcripts[$video_id] = $transcript' \
           "$RESULTS_FILE" > "${RESULTS_FILE}.tmp" && mv "${RESULTS_FILE}.tmp" "$RESULTS_FILE"
    fi
done

# Output results
show_progress "FPL Data Aggregation Complete"

# Console output
echo
echo "=== FPL DATA AGGREGATION SUMMARY ==="
echo "Generated at: $(date)"
echo "Team ID: $TEAM_ID"
echo "Team Name: $TEAM_NAME"  
echo "Total Points: $TOTAL_POINTS"
echo "Current Gameweek: $GAMEWEEK"
echo

echo "=== DATA COLLECTED ==="
echo "• Gameweek Information: ✓"
echo "• Top $PLAYER_COUNT Players by Ownership: ✓" 
echo "• Team Analysis: ✓"
echo "• YouTube Channels Processed: ${#CHANNELS[@]}"
echo "• Videos Discovered: ${#DISCOVERED_VIDEOS[@]}"
echo "• Transcripts Retrieved: $SUCCESSFUL_TRANSCRIPTS"
echo

if [[ ${#DISCOVERED_VIDEOS[@]} -gt 0 ]]; then
    echo "=== DISCOVERED VIDEOS ==="
    for i in "${!DISCOVERY_FILES[@]}"; do
        discovery_file=${DISCOVERY_FILES[$i]}
        channel=${CHANNELS[$i]}
        
        if [[ -f "$discovery_file" && -s "$discovery_file" ]]; then
            video_id=$(jq -r '.video_id // "N/A"' "$discovery_file" 2>/dev/null)
            title=$(jq -r '.title // "N/A"' "$discovery_file" 2>/dev/null)
            confidence=$(jq -r '.confidence // 0' "$discovery_file" 2>/dev/null)
            if [[ "$video_id" != "N/A" && "$video_id" != "null" ]]; then
                echo "• $channel:"
                echo "  - Title: $title"
                echo "  - Video ID: $video_id"
                echo "  - Confidence: $confidence"
            else
                echo "• $channel: No suitable video found"
            fi
        else
            echo "• $channel: Discovery failed"
        fi
    done
    echo
fi

# Write to output file if specified
if [[ -n "$OUTPUT_FILE" ]]; then
    cp "$RESULTS_FILE" "$OUTPUT_FILE"
    log_success "Comprehensive results written to: $OUTPUT_FILE"
    echo "Complete data available in: $OUTPUT_FILE"
else
    echo "=== COMPLETE JSON DATA ==="
    cat "$RESULTS_FILE"
fi

echo
log_success "FPL Data Aggregation completed successfully!"
