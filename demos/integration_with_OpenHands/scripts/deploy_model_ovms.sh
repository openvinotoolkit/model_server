#!/usr/bin/env bash
#
# deploy_model_ovms.sh
#
# Convenience helper for deploying OpenVINO Model Server with OpenHands configuration.
#
# Usage:
#   ./scripts/deploy_model_ovms.sh <model_id> [OPTIONS]
#
# Arguments:
#   model_id          Hugging Face model ID (e.g., "OpenVINO/Qwen3-8b-int8-ov")
#
# Options:
#   --device DEVICE   Target device: CPU or GPU (default: CPU)
#   --cache-dir DIR   Model cache directory (default: ${HOME}/ovms-openhands/models)
#   --skip-wait       Skip health check and return immediately after deploy
#
# Example:
#   ./scripts/deploy_model_ovms.sh OpenVINO/Qwen3-8b-int8-ov --device CPU
#
# Environment Variables:
#   HF_TOKEN          Hugging Face token for gated models (required for some models)
#   LOCAL_NAME        Override the local model name (default: auto-normalized from model_id)
#   MODEL_CACHE_DIR   Override model cache directory
#   TARGET_DEVICE     Override target device
#   OVMS_REST_PORT    OVMS REST API published port (default: 8000)
#   OPENHANDS_PORT    OpenHands Web UI published port (default: 3000)

set -euo pipefail

################################################################################
# Constants and Directory Resolution
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_TEMPLATE="${DEMO_ROOT}/docker-compose.template.yml"
DEFAULT_COMPOSE_FILE="${DEMO_ROOT}/docker-compose.yml"
DEFAULT_MODEL_CACHE_DIR="${HOME}/ovms-openhands/models"
OVMS_CONTAINER_NAME="ovms-llm"

# Configurable ports with defaults
OVMS_REST_PORT="${OVMS_REST_PORT:-8000}"
OPENHANDS_PORT="${OPENHANDS_PORT:-3000}"

################################################################################
# Argument Parsing
################################################################################

print_usage() {
    grep '^#' "${BASH_SOURCE[0]}" | grep -v '^#!/usr/bin/env' | grep -v '^####' | sed 's/^# //' | sed 's/^#//'
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
    MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${DEFAULT_MODEL_CACHE_DIR}}"
    COMPOSE_FILE="${DEFAULT_COMPOSE_FILE}"
    SKIP_WAIT=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --device)
                if [[ -z "${2:-}" ]]; then
                    echo "ERROR: --device requires a value (CPU or GPU)" >&2
                    exit 1
                fi
                TARGET_DEVICE="$2"
                shift 2
                ;;
            --cache-dir)
                if [[ -z "${2:-}" ]]; then
                    echo "ERROR: --cache-dir requires a directory path" >&2
                    exit 1
                fi
                MODEL_CACHE_DIR="$2"
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
        errors=$((errors + 1))
    fi

    # Check envsubst
    if ! command -v envsubst &>/dev/null; then
        echo "ERROR: envsubst is not installed or not in PATH" >&2
        echo "Install gettext package for envsubst support" >&2
        errors=$((errors + 1))
    fi

    # Check HF_TOKEN for gated models (warning only)
    if [[ -z "${HF_TOKEN:-}" ]]; then
        if [[ "$MODEL_ID" =~ meta-llama|Llama|mistralai ]]; then
            echo "WARNING: HF_TOKEN is not set. This model may require authentication." >&2
        fi
    fi

    # Validate compose template exists
    if [[ ! -f "$COMPOSE_TEMPLATE" ]]; then
        echo "ERROR: docker-compose.template.yml not found: $COMPOSE_TEMPLATE" >&2
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
# Model Name Normalization
################################################################################

normalize_model_name() {
    local model_id="$1"

    # Convert Hugging Face model ID to filesystem-safe local name
    # e.g., "OpenVINO/qwen3-8b-int8-ov" -> "qwen3-8b-int8-ov"
    basename "$model_id" | tr '[:upper:]' '[:lower:]' | tr ' ' '-'
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
    # Normalize model name if not overridden
    LOCAL_NAME="${LOCAL_NAME:-$(normalize_model_name "$MODEL_ID")}"

    # Select OVMS Docker image based on target device
    case "$TARGET_DEVICE" in
        GPU)
            OVMS_IMAGE="openvino/model_server:weekly"
            ;;
        *)
            OVMS_IMAGE="openvino/model_server:latest"
            ;;
    esac

    # GPU arguments for docker-compose.yml
    # Matches the pattern used across all demos and the Makefile:
    #   --device /dev/dri --group-add=$(stat -c "%g" /dev/dri/render* | head -n 1)
    if [[ "$TARGET_DEVICE" == "GPU" ]]; then
        GPU_ARGS="    devices:
      - /dev/dri
    group_add:
      - \"$(stat -c '%g' /dev/dri/render* | head -n 1)\""
    else
        GPU_ARGS=""
    fi

    # Host UID/GID for container user mapping
    HOST_UID="$(id -u)"
    HOST_GID="$(id -g)"

    # Export all variables consumed by docker-compose.yml
    export MODEL_ID
    export LOCAL_NAME
    export TARGET_DEVICE
    export MODEL_CACHE_DIR
    export HF_TOKEN="${HF_TOKEN:-}"
    export OVMS_REST_PORT
    export OPENHANDS_PORT
    export OVMS_IMAGE
    export GPU_ARGS
    export HOST_UID
    export HOST_GID

    # Export proxy variables if set (forward to containers)
    [[ -n "${http_proxy:-}" ]] && export http_proxy
    [[ -n "${https_proxy:-}" ]] && export https_proxy
    [[ -n "${HTTP_PROXY:-}" ]] && export HTTP_PROXY
    [[ -n "${HTTPS_PROXY:-}" ]] && export HTTPS_PROXY
    [[ -n "${no_proxy:-}" ]] && export no_proxy
    [[ -n "${NO_PROXY:-}" ]] && export NO_PROXY

    echo "Runtime configuration:"
    echo "  MODEL_ID:          $MODEL_ID"
    echo "  LOCAL_NAME:         $LOCAL_NAME"
    echo "  TARGET_DEVICE:      $TARGET_DEVICE"
    echo "  MODEL_CACHE_DIR:    $MODEL_CACHE_DIR"
    echo "  HF_TOKEN:           ${HF_TOKEN:+<set>}"
    echo "  OVMS_IMAGE:         $OVMS_IMAGE"
    echo "  OVMS_REST_PORT:     $OVMS_REST_PORT"
    echo "  OPENHANDS_PORT:     $OPENHANDS_PORT"
}

################################################################################
# Docker Compose Deployment
################################################################################

generate_compose_from_template() {
    local template_file="$1"
    local output_file="$2"

    echo "Generating docker-compose.yml from template..."

    # envsubst with explicit variable allowlist
    envsubst '$MODEL_ID $LOCAL_NAME $TARGET_DEVICE $MODEL_CACHE_DIR $HF_TOKEN $OVMS_IMAGE $GPU_ARGS $HOST_UID $HOST_GID $OVMS_REST_PORT $OPENHANDS_PORT $http_proxy $https_proxy $HTTP_PROXY $HTTPS_PROXY $no_proxy $NO_PROXY' < "$template_file" > "$output_file"
    echo "  Generated: $output_file"
}

deploy_ovms() {
    local compose_file="$1"
    local template_file="$COMPOSE_TEMPLATE"

    echo "Deploying OVMS and OpenHands via Docker Compose..."

    # Always (re)generate from template for simplicity
    generate_compose_from_template "$template_file" "$compose_file"

    # Deploy via docker compose
    docker compose -f "$compose_file" up -d
}

################################################################################
# Health Check Polling
################################################################################

wait_for_health() {
    local max_retries=60  # 60 * 5s = 5 minutes total polling
    local retry_interval=5

    echo "Waiting for OVMS LLM graph to initialize"

    # Initial sleep to let container start
    sleep 10

    for i in $(seq 1 $max_retries); do
        local status
        status=$(curl -sf "http://localhost:${OVMS_REST_PORT}/v1/config" 2>/dev/null || true)

        if echo "$status" | grep -q '"AVAILABLE"'; then
            echo "OVMS is ready. Model status: AVAILABLE"
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
# Error Handling
################################################################################

cleanup_on_error() {
    echo ""
    echo "Deployment failed. Check container logs:"
    echo "  docker logs $OVMS_CONTAINER_NAME"
    echo ""
}

################################################################################
# Main Orchestration
################################################################################

main() {
    parse_args "$@"
    validate_prerequisites
    validate_device "$TARGET_DEVICE"

    # Prepare workspace
    prepare_model_workspace "$MODEL_CACHE_DIR"

    # Export runtime configuration
    if ! export_runtime_configuration; then
        exit 1
    fi

    # Deploy
    deploy_ovms "$COMPOSE_FILE"

    # Health check (unless skipped)
    if [[ "$SKIP_WAIT" == "false" ]]; then
        if ! wait_for_health; then
            cleanup_on_error
            exit 1
        fi
    else
        echo "Skipping health check (--skip-wait specified)"
    fi

    # Success summary
    echo "Deployment complete!"
    echo ""
    echo "Services running:"
    echo "  - OVMS:        http://localhost:${OVMS_REST_PORT}/v3"
    echo "  - OpenHands:   http://localhost:${OPENHANDS_PORT}"
    echo ""
    echo "Next steps:"
    echo "  1. Verify OVMS: curl http://localhost:${OVMS_REST_PORT}/v3/models"
    echo "  2. Open OpenHands: http://localhost:${OPENHANDS_PORT}"
    echo "  3. Create an agent task to test the integration"
    echo ""
}

# Execute main function with all arguments
main "$@"