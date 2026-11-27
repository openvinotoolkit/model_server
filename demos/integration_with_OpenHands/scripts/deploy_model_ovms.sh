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
#   --device DEVICE   Target device: CPU or GPU (default: automatic)
#   --cache-dir DIR   Model cache directory (default: ${HOME}/ovms-openhands/models)
#   --skip-wait       Skip waiting for OVMS model health
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

# --- Constants and Directory Resolution ---

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_TEMPLATE="${DEMO_ROOT}/docker-compose.template.yml"
COMPOSE_FILE="${DEMO_ROOT}/docker-compose.yml"
DEFAULT_MODEL_CACHE_DIR="${HOME}/ovms-openhands/models"
OVMS_CONTAINER_NAME="ovms-llm"

# Configurable ports with defaults
OVMS_REST_PORT="${OVMS_REST_PORT:-8000}"
OPENHANDS_PORT="${OPENHANDS_PORT:-3000}"

# --- Argument Parsing ---

print_usage() {
    sed -n '1,29p' "${BASH_SOURCE[0]}" | grep '^#' | grep -v '^#!/usr/bin/env' | sed 's/^# //' | sed 's/^#//'
    exit 0
}

if [[ $# -eq 0 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    print_usage
fi

MODEL_ID="$1"
shift

# Initialize from environment or defaults
TARGET_DEVICE="${TARGET_DEVICE:-}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-${DEFAULT_MODEL_CACHE_DIR}}"
SKIP_WAIT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            [[ -z "${2:-}" ]] && { echo "ERROR: --device requires a value (CPU or GPU)" >&2; exit 1; }
            TARGET_DEVICE="$2"
            shift 2
            ;;
        --cache-dir)
            [[ -z "${2:-}" ]] && { echo "ERROR: --cache-dir requires a directory path" >&2; exit 1; }
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

# --- Validation ---

validate_device() {
    local device="$1"
    [[ -z "$device" ]] && return 0
    case "$device" in
        CPU|GPU) ;;
        *) echo "ERROR: Invalid device: $device" >&2; echo "Supported devices: CPU, GPU" >&2; exit 1 ;;
    esac

    if [[ "$device" == "GPU" ]]; then
        local render_devices=(/dev/dri/render*)
        if [[ ! -e "${render_devices[0]}" ]]; then
            echo "ERROR: GPU was requested, but no /dev/dri/render device is available" >&2
            exit 1
        fi
    fi
}

validate_port() {
    local name="$1"
    local port="$2"

    if ! [[ "$port" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
        echo "ERROR: $name must be an integer between 1 and 65535: $port" >&2
        exit 1
    fi
}

validate_port_available() {
    local name="$1"
    local port="$2"
    local container="$3"

    command -v ss &>/dev/null || return 0
    if ! ss -H -ltn "sport = :${port}" | grep -q .; then
        return 0
    fi

    local published_port
    published_port=$(docker inspect --format \
        "{{range .NetworkSettings.Ports}}{{range .}}{{if eq .HostPort \"${port}\"}}{{println .HostPort}}{{end}}{{end}}{{end}}" \
        "$container" 2>/dev/null || true)
    if [[ -n "$published_port" ]]; then
        return 0
    fi

    echo "ERROR: $name port $port is already in use." >&2
    echo "Choose an unused port, for example: $name=$((port + 1)) ./scripts/deploy_model_ovms.sh $MODEL_ID --device ${TARGET_DEVICE:-CPU}" >&2
    exit 1
}

validate_prerequisites() {
    local errors=0

    command -v docker &>/dev/null || { echo "ERROR: Docker is not installed or not in PATH" >&2; errors=$((errors + 1)); }
    docker compose version &>/dev/null || { echo "ERROR: docker compose plugin is not available" >&2; errors=$((errors + 1)); }
    command -v envsubst &>/dev/null || { echo "ERROR: envsubst is not installed (install gettext)" >&2; errors=$((errors + 1)); }

    if [[ -z "${HF_TOKEN:-}" ]] && [[ "$MODEL_ID" =~ meta-llama|Llama|mistralai ]]; then
        echo "WARNING: HF_TOKEN is not set. This model may require authentication." >&2
    fi

    [[ -f "$COMPOSE_TEMPLATE" ]] || { echo "ERROR: docker-compose.template.yml not found: $COMPOSE_TEMPLATE" >&2; errors=$((errors + 1)); }

    return $errors
}

# --- Model Name Normalization ---

normalize_model_name() {
    basename "$1" | tr '[:upper:]' '[:lower:]' | tr ' ' '-'
}

# --- Runtime Configuration Export ---

export_runtime_configuration() {
    LOCAL_NAME="${LOCAL_NAME:-$(normalize_model_name "$MODEL_ID")}"
    OVMS_IMAGE="${OVMS_IMAGE:-openvino/model_server:weekly}"

    GPU_ARGS=""
    if [[ "$TARGET_DEVICE" == "GPU" ]]; then
        local render_devices=(/dev/dri/render*)
        if [[ -e "${render_devices[0]}" ]]; then
            GPU_ARGS="    devices:
      - /dev/dri
    group_add:
      - \"$(stat -c '%g' /dev/dri/render* | head -n 1)\""
        fi
    fi

    HOST_UID="$(id -u)"
    HOST_GID="$(id -g)"

    # Export all variables consumed by the Compose template.
    # Every variable referenced in the template must be exported (even if empty)
    # so that envsubst replaces it cleanly instead of leaving a literal ${VAR}.
    export MODEL_ID LOCAL_NAME TARGET_DEVICE MODEL_CACHE_DIR
    export OVMS_REST_PORT OPENHANDS_PORT OVMS_IMAGE GPU_ARGS HOST_UID HOST_GID

    # Token and proxy variables: default to empty so envsubst produces blank values
    # rather than leaving shell default expressions in the generated file.
    export HF_TOKEN="${HF_TOKEN:-}"
    export http_proxy="${http_proxy:-}"
    export https_proxy="${https_proxy:-}"
    export HTTP_PROXY="${HTTP_PROXY:-}"
    export HTTPS_PROXY="${HTTPS_PROXY:-}"
    export no_proxy="${no_proxy:-}"
    export NO_PROXY="${NO_PROXY:-}"
    export OPENHANDS_NO_PROXY="${no_proxy:+${no_proxy},}localhost,127.0.0.1,ovms-llm"
    export OPENHANDS_NO_PROXY_UPPER="${NO_PROXY:+${NO_PROXY},}localhost,127.0.0.1,ovms-llm"

    echo "Runtime configuration:"
    echo "  MODEL_ID:       $MODEL_ID"
    echo "  LOCAL_NAME:     $LOCAL_NAME"
    echo "  TARGET_DEVICE:  $TARGET_DEVICE"
    echo "  MODEL_CACHE_DIR:$MODEL_CACHE_DIR"
    echo "  HF_TOKEN:       ${HF_TOKEN:+<set>}"
    echo "  OVMS_IMAGE:     $OVMS_IMAGE"
    echo "  OVMS_REST_PORT: $OVMS_REST_PORT"
    echo "  OPENHANDS_PORT: $OPENHANDS_PORT"
}

# --- OpenHands Initial Settings ---

initialize_openhands_settings() {
    local container="${1:-openhands}"
    local attempt
    local ready=false

    echo "Waiting for OpenHands Agent Server..."
    for attempt in $(seq 1 90); do
        if docker exec "$container" sh -c 'curl --noproxy "*" -fsS http://127.0.0.1:18000/alive >/dev/null' 2>/dev/null; then
            ready=true
            break
        fi
        sleep 2
    done
    if [[ "$ready" != "true" ]]; then
        echo "ERROR: OpenHands Agent Server did not become ready. Check: docker logs $container" >&2
        return 1
    fi

    docker exec "$container" sh -c '
        set -eu
        key=$(cat /home/openhands/.openhands/agent-canvas/api-key.txt)
        api=http://127.0.0.1:18000/api/profiles
        profiles=$(curl --noproxy "*" -fsS -H "X-Session-API-Key: $key" "$api")
        count=$(printf "%s" "$profiles" | jq -er ".profiles | length")
        if [ "$count" -ne 0 ]; then
            echo "Keeping existing OpenHands LLM profiles."
            exit 0
        fi
        body=$(jq -n --arg model "openai/$1" \
            "{llm:{model:\$model,base_url:\"http://ovms-llm:8000/v3\",api_key:\"unused\",native_tool_calling:true,max_output_tokens:5000,temperature:0}}")
        curl --noproxy "*" -fsS -o /dev/null -H "X-Session-API-Key: $key" \
            -H "Content-Type: application/json" -d "$body" "$api/ovms"
        curl --noproxy "*" -fsS -o /dev/null -X POST -H "X-Session-API-Key: $key" "$api/ovms/activate"
        echo "Configured and activated the OVMS LLM profile."
    ' sh "$LOCAL_NAME"
}

# --- Docker Compose Deployment ---

deploy_ovms() {
    echo "Generating docker-compose.yml from template..."
    envsubst '$MODEL_ID $LOCAL_NAME $TARGET_DEVICE $MODEL_CACHE_DIR $HF_TOKEN
        $OVMS_IMAGE $GPU_ARGS $HOST_UID $HOST_GID $OVMS_REST_PORT $OPENHANDS_PORT
        $http_proxy $https_proxy $HTTP_PROXY $HTTPS_PROXY $no_proxy $NO_PROXY
        $OPENHANDS_NO_PROXY $OPENHANDS_NO_PROXY_UPPER' \
        < "$COMPOSE_TEMPLATE" > "$COMPOSE_FILE"
    chmod 600 "$COMPOSE_FILE"
    echo "  Generated: $COMPOSE_FILE"

    echo "Deploying OVMS and OpenHands via Docker Compose..."
    if [[ "$SKIP_WAIT" == "true" ]]; then
        docker compose -f "$COMPOSE_FILE" up -d --no-deps ovms-llm
        docker compose -f "$COMPOSE_FILE" up -d --no-deps openhands
    else
        docker compose -f "$COMPOSE_FILE" up -d
    fi
}

# --- Health Check ---

wait_for_health() {
    local max_retries=60  # 60 * 5s = 5 minutes
    local retry_interval=5

    echo "Waiting for OVMS LLM graph to initialize"
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

# --- Main Orchestration ---

main() {
    validate_device "$TARGET_DEVICE"
    validate_port "OVMS_REST_PORT" "$OVMS_REST_PORT"
    validate_port "OPENHANDS_PORT" "$OPENHANDS_PORT"
    validate_prerequisites

    [[ "$OVMS_REST_PORT" != "$OPENHANDS_PORT" ]] || {
        echo "ERROR: OVMS_REST_PORT and OPENHANDS_PORT must be different" >&2
        exit 1
    }

    validate_port_available "OVMS_REST_PORT" "$OVMS_REST_PORT" "ovms-llm"
    validate_port_available "OPENHANDS_PORT" "$OPENHANDS_PORT" "openhands"

    mkdir -p "$MODEL_CACHE_DIR"

    echo "Using model cache directory: $MODEL_CACHE_DIR"

    # Normalize to absolute path so Docker mounts the intended host directory
    # regardless of which working directory the script is invoked from.
    MODEL_CACHE_DIR="$(cd "$MODEL_CACHE_DIR" && pwd)" || {
        echo "ERROR: Cannot resolve model cache directory: $MODEL_CACHE_DIR" >&2
        exit 1
    }
    echo "Model cache ready: $MODEL_CACHE_DIR"

    export_runtime_configuration
    deploy_ovms
    initialize_openhands_settings

    if [[ "$SKIP_WAIT" == "false" ]]; then
        if ! wait_for_health; then
            echo ""
            echo "Deployment failed. Check container logs:"
            echo "  docker logs $OVMS_CONTAINER_NAME"
            echo ""
            exit 1
        fi
    else
        echo "Skipping health check (--skip-wait specified)"
    fi

    echo "Deployment complete!"
    echo ""
    echo "Services running:"
    echo "  - OVMS:        http://localhost:${OVMS_REST_PORT}/v3"
    echo "  - OpenHands:   http://localhost:${OPENHANDS_PORT}/canvas"
    echo ""
    echo "Next steps:"
    echo "  1. Verify OVMS: curl http://localhost:${OVMS_REST_PORT}/v3/models"
    echo "  2. Open OpenHands: http://localhost:${OPENHANDS_PORT}/canvas"
    echo "  3. Check the active LLM profile and create an agent task"
    echo ""
}

main "$@"