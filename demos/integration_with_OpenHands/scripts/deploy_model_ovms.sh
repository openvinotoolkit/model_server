#!/usr/bin/env bash
#
# deploy_model_ovms.sh
#
# Convenience helper for deploying OpenVINO Model Server with OpenHands configuration.
#
# This script automates the runtime environment setup and OVMS deployment documented
# in the README.md. It is optional - users can achieve the same result by following
# the manual workflow documented in the README.
#
# Usage:
#   ./scripts/deploy_model_ovms.sh <model_id> [OPTIONS]
#
# Arguments:
#   model_id          Hugging Face model ID (e.g., "OpenVINO/qwen3-0.6b-int8-ov")
#
# Options:
#   --device DEVICE       Target device: CPU or GPU (default: CPU)
#   --parser PARSER       Tool parser: hermes3, qwen, or none (default: auto-resolved)
#   --cache-dir DIR       Model cache directory (default: ${HOME}/ovms-openhands/models)
#   --compose-file FILE   Path to docker-compose.yml (default: <demo_root>/docker-compose.yml)
#   --skip-wait           Skip health check and return immediately after deploy
#
# Example:
#   ./scripts/deploy_model_ovms.sh OpenVINO/qwen3-0.6b-int8-ov --device CPU
#
# Environment Variables:
#   HF_TOKEN          Hugging Face token for gated models (required for some models)
#   LOCAL_NAME        Override the local model name (default: auto-normalized from model_id)
#   MODEL_CACHE_DIR   Override model cache directory
#   TARGET_DEVICE     Override target device
#   TOOL_PARSER       Override tool parser
#
# The script exports environment variables consumed by docker-compose.yml:
#   MODEL_ID, LOCAL_NAME, TARGET_DEVICE, TOOL_PARSER, MODEL_CACHE_DIR

set -euo pipefail

################################################################################
# Constants and Directory Resolution
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_COMPOSE_FILE="${DEMO_ROOT}/docker-compose.yml"
DEFAULT_MODEL_CACHE_DIR="${HOME}/ovms-openhands/models"
OVMS_CONTAINER_NAME="ovms-llm"
OPENHANDS_CONTAINER_NAME="openhands"
DOCKER_NETWORK="ovms-net"
OVMS_REST_PORT="8000"

# Tool parser mapping: model family patterns to parser names
declare -A TOOL_PARSERS=(
    ["Qwen"]="hermes3"
    ["qwen"]="hermes3"
    ["Llama3"]="hermes3"
    ["llama3"]="hermes3"
    ["Mistral"]="hermes3"
    ["mistral"]="hermes3"
)

################################################################################
# Argument Parsing
################################################################################

print_usage() {
    grep '^#' "${BASH_SOURCE[0]}" | grep -v '^#!/usr/bin/env' | sed 's/^# //' | sed 's/^#//'
    exit 0
}

parse_args() {
    if [[ $# -eq 0 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        print_usage
    fi

    MODEL_ID="$1"
    shift

    # Initialize from environment or defaults
    TARGET_DEVICE="${TARGET_DEVICE:-CPU}"
    TOOL_PARSER="${TOOL_PARSER:-}"
    MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${DEFAULT_MODEL_CACHE_DIR}}"
    COMPOSE_FILE="${DEFAULT_COMPOSE_FILE}"
    SKIP_WAIT=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --device)
                TARGET_DEVICE="$2"
                shift 2
                ;;
            --parser)
                TOOL_PARSER="$2"
                shift 2
                ;;
            --cache-dir)
                MODEL_CACHE_DIR="$2"
                shift 2
                ;;
            --compose-file)
                COMPOSE_FILE="$2"
                shift 2
                ;;
            --skip-wait)
                SKIP_WAIT=true
                shift
                ;;
            *)
                echo "ERROR: Unknown option: $1" >&2
                echo "Use --help for usage information." >&2
                exit 1
                ;;
        esac
    done
}

################################################################################
# Validation Functions
################################################################################

validate_prerequisites() {
    local errors=0

    # Check Docker
    if ! command -v docker &>/dev/null; then
        echo "ERROR: Docker is not installed or not in PATH" >&2
        errors=$((errors + 1))
    fi

    # Check Docker Compose plugin
    if ! docker compose version &>/dev/null; then
        echo "ERROR: docker compose plugin is not available" >&2
        echo "Install Docker Compose v2 or use 'docker-compose' standalone" >&2
        errors=$((errors + 1))
    fi

    # Check HF_TOKEN for gated models (warning only)
    if [[ -z "${HF_TOKEN:-}" ]]; then
        if [[ "$MODEL_ID" =~ meta-llama|Llama|mistralai ]]; then
            echo "WARNING: HF_TOKEN is not set. This model may require authentication." >&2
            echo "Set HF_TOKEN environment variable for gated models." >&2
        fi
    fi

    # Validate compose file
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        echo "ERROR: docker-compose.yml not found: $COMPOSE_FILE" >&2
        errors=$((errors + 1))
    fi

    return $errors
}

validate_device() {
    local device="$1"

    case "$device" in
        CPU|GPU)
            # Valid device types
            ;;
        *)
            echo "ERROR: Invalid device: $device" >&2
            echo "Supported devices: CPU, GPU" >&2
            exit 1
            ;;
    esac
}

################################################################################
# Model Name Normalization and Tool Parser Resolution
################################################################################

normalize_model_name() {
    local model_id="$1"

    # Convert Hugging Face model ID to filesystem-safe local name
    # e.g., "OpenVINO/qwen3-0.6b-int8-ov" → "qwen3-0.6b-int8-ov"
    basename "$model_id" | tr '[:upper:]' '[:lower:]' | tr ' ' '-'
}

resolve_tool_parser() {
    local model_id="$1"
    local override="${2:-}"

    # If override provided, use it
    if [[ -n "$override" ]]; then
        echo "$override"
        return
    fi

    # Try to match against known model families
    for pattern in "${!TOOL_PARSERS[@]}"; do
        if [[ "$model_id" == *"$pattern"* ]]; then
            echo "${TOOL_PARSERS[$pattern]}"
            return
        fi
    done

    # Default: no tool parser
    echo "none"
}

################################################################################
# Workspace Preparation
################################################################################

prepare_model_workspace() {
    local cache_dir="$1"

    # Create cache directory if it doesn't exist
    if [[ ! -d "$cache_dir" ]]; then
        echo "Creating model cache directory: $cache_dir"
        mkdir -p "$cache_dir"
    fi

    # OVMS will handle model download and graph generation via --source_model
    echo "Model cache ready: $cache_dir"
}

################################################################################
# Runtime Configuration Export
################################################################################

export_runtime_configuration() {
    # Export all variables consumed by docker-compose.yml
    export MODEL_ID
    export LOCAL_NAME
    export TARGET_DEVICE
    export TOOL_PARSER
    export MODEL_CACHE_DIR
    export HF_TOKEN="${HF_TOKEN:-}"

    echo "Runtime configuration:"
    echo "  MODEL_ID:        $MODEL_ID"
    echo "  LOCAL_NAME:       $LOCAL_NAME"
    echo "  TARGET_DEVICE:    $TARGET_DEVICE"
    echo "  TOOL_PARSER:      $TOOL_PARSER"
    echo "  MODEL_CACHE_DIR:  $MODEL_CACHE_DIR"
    echo "  HF_TOKEN:         ${HF_TOKEN:+<set>}"
}

################################################################################
# Docker Compose Deployment
################################################################################

deploy_ovms() {
    local compose_file="$1"

    echo "Deploying OVMS and OpenHands via Docker Compose..."

    # Stop existing containers if running
    if docker ps -a --format '{{.Names}}' | grep -q "^${OVMS_CONTAINER_NAME}$"; then
        echo "Stopping existing OVMS container: $OVMS_CONTAINER_NAME"
        docker stop "$OVMS_CONTAINER_NAME" >/dev/null 2>&1 || true
        docker rm "$OVMS_CONTAINER_NAME" >/dev/null 2>&1 || true
    fi

    if docker ps -a --format '{{.Names}}' | grep -q "^${OPENHANDS_CONTAINER_NAME}$"; then
        echo "Stopping existing OpenHands container: $OPENHANDS_CONTAINER_NAME"
        docker stop "$OPENHANDS_CONTAINER_NAME" >/dev/null 2>&1 || true
        docker rm "$OPENHANDS_CONTAINER_NAME" >/dev/null 2>&1 || true
    fi

    # Deploy via docker compose
    docker compose -f "$compose_file" up -d
}

################################################################################
# Health Check Polling
################################################################################

wait_for_health() {
    local max_retries=36  # 36 * 100s = 60 minutes
    local retry_interval=100

    echo "Waiting for OVMS LLM graph to initialize (this may take 30-60 seconds)..."

    # Initial sleep to let container start
    sleep 10

    for i in $(seq 1 $max_retries); do
        local status
        status=$(curl -sf "http://localhost:${OVMS_REST_PORT}/v1/config" 2>/dev/null || true)

        if echo "$status" | grep -q '"AVAILABLE"'; then
            echo "✓ OVMS is ready. Model status: AVAILABLE"
            return 0
        fi

        echo "  Attempt $i/$max_retries: model not available yet..."
        sleep "$retry_interval"
    done

    echo "ERROR: OVMS failed to become ready within expected time." >&2
    echo "Check container logs: docker logs $OVMS_CONTAINER_NAME" >&2
    return 1
}

################################################################################
# Diagnostics and Manual Equivalent
################################################################################

print_manual_equivalent() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Manual Equivalent (README documents this workflow)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "The same deployment can be achieved manually with:"
    echo ""
    echo "  # Set environment variables"
    echo "  export MODEL_ID=\"$MODEL_ID\""
    echo "  export LOCAL_NAME=\"$LOCAL_NAME\""
    echo "  export TARGET_DEVICE=\"$TARGET_DEVICE\""
    echo "  export TOOL_PARSER=\"$TOOL_PARSER\""
    echo "  export MODEL_CACHE_DIR=\"$MODEL_CACHE_DIR"
    echo "  export HF_TOKEN=\"\${HF_TOKEN:-}\""
    echo ""
    echo "  # Deploy via Docker Compose"
    echo "  docker compose -f $COMPOSE_FILE up -d"
    echo ""
    echo "  # Wait for OVMS to become ready"
    echo "  curl -sf http://localhost:8000/v1/config | grep AVAILABLE"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

cleanup_on_error() {
    local exit_code="$1"

    if [[ $exit_code -ne 0 ]]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Deployment failed. For troubleshooting, see:"
        echo "  - Container logs: docker logs $OVMS_CONTAINER_NAME"
        echo "  - README.md troubleshooting section"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
    fi
}

################################################################################
# Main Orchestration
################################################################################

main() {
    parse_args "$@"
    validate_prerequisites
    validate_device "$TARGET_DEVICE"

    # Normalize model name if not overridden
    LOCAL_NAME="${LOCAL_NAME:-$(normalize_model_name "$MODEL_ID")}"

    # Resolve tool parser if not overridden
    TOOL_PARSER="$(resolve_tool_parser "$MODEL_ID" "$TOOL_PARSER")"

    # Prepare workspace
    prepare_model_workspace "$MODEL_CACHE_DIR"

    # Export runtime configuration
    export_runtime_configuration

    # Deploy
    deploy_ovms "$COMPOSE_FILE"

    # Health check (unless skipped)
    if [[ "$SKIP_WAIT" == "false" ]]; then
        if ! wait_for_health; then
            cleanup_on_error 1
            exit 1
        fi
    else
        echo "Skipping health check (--skip-wait specified)"
    fi

    # Print manual equivalent
    print_manual_equivalent

    # Success summary
    echo "✓ Deployment complete!"
    echo ""
    echo "Services running:"
    echo "  - OVMS:        http://localhost:${OVMS_REST_PORT}/v3"
    echo "  - OpenHands:   http://localhost:3000"
    echo ""
    echo "Next steps (from README.md):"
    echo "  1. Verify OVMS: curl http://localhost:${OVMS_REST_PORT}/v3/models"
    echo "  2. Open OpenHands: http://localhost:3000"
    echo "  3. Create an agent task to test the integration"
    echo ""
}

# Execute main function with all arguments
main "$@"
