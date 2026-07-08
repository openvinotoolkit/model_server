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
# Compose file workflow:
#   - docker-compose.template.yml is the immutable template (tracked in Git)
#   - docker-compose.yml is generated from the template (not tracked)
#   - .ovms-deployment stores the complete deployment fingerprint
#   - The script compares the deployment fingerprint:
#     * Identical fingerprint: preserves existing compose (user edits kept)
#     * Different fingerprint: regenerates compose (discards old edits)
#   - Users may freely edit the generated docker-compose.yml after deployment
#
# Usage:
#   ./scripts/deploy_model_ovms.sh <model_id> [OPTIONS]
#
# Arguments:
#   model_id          Hugging Face model ID (e.g., "OpenVINO/qwen3-0.6b-int8-ov")
#
# Options:
#   --device DEVICE           Target device: CPU or GPU (default: CPU)
#   --parser PARSER           Tool parser: hermes3, qwen3coder, devstral, gemma4, gptoss, llama3, mistral, phi4, or none (default: auto-resolved)
#   --reasoning-parser PARSER Reasoning parser: gemma4, gptoss, or none (default: auto-resolved)
#   --cache-dir DIR           Model cache directory (default: ${HOME}/ovms-openhands/models)
#   --compose-file FILE       Path to docker-compose.yml (default: <demo_root>/docker-compose.yml, generated from template)
#   --skip-wait               Skip health check and return immediately after deploy
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
#   REASONING_PARSER  Override reasoning parser
#   OVMS_REST_PORT    OVMS REST API published port (default: 8000)
#   OVMS_GRPC_PORT    OVMS gRPC API published port (default: 9000)
#   OPENHANDS_PORT    OpenHands Web UI published port (default: 3000)
#   http_proxy        HTTP proxy for container network access
#   https_proxy       HTTPS proxy for container network access
#   HTTP_PROXY        HTTP proxy (uppercase variant)
#   HTTPS_PROXY       HTTPS proxy (uppercase variant)
#   no_proxy          No-proxy list for container network access
#   NO_PROXY          No-proxy list (uppercase variant)
#
# The script exports environment variables consumed by docker-compose.yml:
#   MODEL_ID, LOCAL_NAME, TARGET_DEVICE, TOOL_PARSER, REASONING_PARSER, MODEL_CACHE_DIR, GPU_DEVICE, WSL_LIBS
#   OVMS_REST_PORT, OVMS_GRPC_PORT, OPENHANDS_PORT
#   http_proxy, https_proxy, HTTP_PROXY, HTTPS_PROXY, no_proxy, NO_PROXY

set -euo pipefail

################################################################################
# Constants and Directory Resolution
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_TEMPLATE="${DEMO_ROOT}/docker-compose.template.yml"
DEFAULT_COMPOSE_FILE="${DEMO_ROOT}/docker-compose.yml"
DEPLOYMENT_METADATA="${DEMO_ROOT}/.ovms-deployment"
DEFAULT_MODEL_CACHE_DIR="${HOME}/ovms-openhands/models"
OVMS_CONTAINER_NAME="ovms-llm"
OPENHANDS_CONTAINER_NAME="openhands"
DOCKER_NETWORK="ovms-net"

# Configurable ports with defaults (can be overridden via environment variables)
OVMS_REST_PORT="${OVMS_REST_PORT:-8000}"
OVMS_GRPC_PORT="${OVMS_GRPC_PORT:-9000}"
OPENHANDS_PORT="${OPENHANDS_PORT:-3000}"

# Tool parser mapping: model family patterns to parser names
declare -A TOOL_PARSERS=(
    ["Qwen3"]="hermes3"
    ["qwen3"]="hermes3"
    ["Qwen3-Coder"]="qwen3coder"
    ["qwen3-coder"]="qwen3coder"
    ["Llama3"]="llama3"
    ["llama3"]="llama3"
    ["Mistral"]="mistral"
    ["mistral"]="mistral"
    ["Phi4"]="phi4"
    ["phi4"]="phi4"
    ["Devstral"]="devstral"
    ["devstral"]="devstral"
    ["Gemma4"]="gemma4"
    ["gemma4"]="gemma4"
    ["Gemma-4"]="gemma4"
    ["gemma-4"]="gemma4"
    ["GPT-OSS"]="gptoss"
    ["gpt-oss"]="gptoss"
)

# Reasoning parser mapping: model family patterns to parser names
declare -A REASONING_PARSERS=(
    ["Gemma4"]="gemma4"
    ["gemma4"]="gemma4"
    ["Gemma-4"]="gemma4"
    ["gemma-4"]="gemma4"
    ["GPT-OSS"]="gptoss"
    ["gpt-oss"]="gptoss"
)

################################################################################
# GPU Device Detection
################################################################################

detect_gpu_device() {
    # Check for WSL2 first (GPU passthrough device)
    if [[ -e /dev/dxg ]]; then
        echo "/dev/dxg:/dev/dxg"
    # Fallback: check /proc/version for WSL signature
    elif grep -qi microsoft /proc/version 2>/dev/null; then
        echo "/dev/dxg:/dev/dxg"
    # Native Linux with GPU device
    elif [[ -e /dev/dri ]]; then
        echo "/dev/dri:/dev/dri"
    # No GPU device detected
    else
        echo ""
    fi
}

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
    REASONING_PARSER="${REASONING_PARSER:-}"
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
            --reasoning-parser)
                REASONING_PARSER="$2"
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

resolve_reasoning_parser() {
    local model_id="$1"
    local override="${2:-}"

    # If override provided, use it
    if [[ -n "$override" ]]; then
        echo "$override"
        return
    fi

    # Try to match against known model families
    for pattern in "${!REASONING_PARSERS[@]}"; do
        if [[ "$model_id" == *"$pattern"* ]]; then
            echo "${REASONING_PARSERS[$pattern]}"
            return
        fi
    done

    # Default: no reasoning parser
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
    export REASONING_PARSER
    export MODEL_CACHE_DIR
    export HF_TOKEN="${HF_TOKEN:-}"

    # Export configurable ports
    export OVMS_REST_PORT
    export OVMS_GRPC_PORT
    export OPENHANDS_PORT

    # Export proxy variables if set (forward to containers)
    if [[ -n "${http_proxy:-}" ]]; then
        export http_proxy
    fi
    if [[ -n "${https_proxy:-}" ]]; then
        export https_proxy
    fi
    if [[ -n "${HTTP_PROXY:-}" ]]; then
        export HTTP_PROXY
    fi
    if [[ -n "${HTTPS_PROXY:-}" ]]; then
        export HTTPS_PROXY
    fi
    if [[ -n "${no_proxy:-}" ]]; then
        export no_proxy
    fi
    if [[ -n "${NO_PROXY:-}" ]]; then
        export NO_PROXY
    fi

    # Select OVMS Docker image based on target device
    case "$TARGET_DEVICE" in
        GPU)
            export OVMS_IMAGE="openvino/model_server:latest-gpu"
            ;;
        *)
            export OVMS_IMAGE="openvino/model_server:latest"
            ;;
    esac

    # Detect and export GPU device for docker-compose.yml devices mapping
    export GPU_DEVICE
    GPU_DEVICE="$(detect_gpu_device)"

    # WSL library dependencies (only needed for WSL2 GPU passthrough)
    if [[ "$GPU_DEVICE" == *"/dev/dxg"* ]]; then
        export WSL_LIBS="/usr/lib/wsl:/usr/lib/wsl:ro"
    else
        export WSL_LIBS=""
    fi

    echo "Runtime configuration:"
    echo "  MODEL_ID:          $MODEL_ID"
    echo "  LOCAL_NAME:         $LOCAL_NAME"
    echo "  TARGET_DEVICE:      $TARGET_DEVICE"
    echo "  TOOL_PARSER:        $TOOL_PARSER"
    echo "  REASONING_PARSER:   $REASONING_PARSER"
    echo "  MODEL_CACHE_DIR:    $MODEL_CACHE_DIR"
    echo "  HF_TOKEN:           ${HF_TOKEN:+<set>}"
    echo "  OVMS_REST_PORT:     $OVMS_REST_PORT"
    echo "  OVMS_GRPC_PORT:     $OVMS_GRPC_PORT"
    echo "  OPENHANDS_PORT:     $OPENHANDS_PORT"
}

################################################################################
# Compose File Generation and Management
################################################################################

# Metadata file management
# .ovms-deployment stores the complete deployment fingerprint for comparison

write_deployment_metadata() {
    local metadata_file="$1"
    local timestamp="$2"

    cat > "$metadata_file" << EOF
METADATA_VERSION=1
MODEL_ID=${MODEL_ID}
TARGET_DEVICE=${TARGET_DEVICE}
LOCAL_NAME=${LOCAL_NAME}
TOOL_PARSER=${TOOL_PARSER}
REASONING_PARSER=${REASONING_PARSER}
OVMS_IMAGE=${OVMS_IMAGE}
GPU_DEVICE=${GPU_DEVICE}
MODEL_CACHE_DIR=${MODEL_CACHE_DIR}
OVMS_REST_PORT=${OVMS_REST_PORT}
OVMS_GRPC_PORT=${OVMS_GRPC_PORT}
OPENHANDS_PORT=${OPENHANDS_PORT}
http_proxy=${http_proxy:-}
https_proxy=${https_proxy:-}
HTTP_PROXY=${HTTP_PROXY:-}
HTTPS_PROXY=${HTTPS_PROXY:-}
no_proxy=${no_proxy:-}
NO_PROXY=${NO_PROXY:-}
GENERATION_TIMESTAMP=${timestamp}
EOF
}

# Compare current deployment fingerprint with stored metadata
# Returns 0 if all deployment parameters match, 1 if any differ
# Sets METADATA_OUTDATED to 1 if metadata version is incompatible
compare_deployment_fingerprint() {
    local metadata_file="$1"

    if [[ ! -f "$metadata_file" ]]; then
        return 1
    fi

    # Export current runtime values with CURRENT_ prefix for comparison
    # Source metadata in a subshell to avoid overwriting global variables
    # The subshell outputs a status code and outdated flag
    local result
    result=$(
        # Current runtime values (from caller's scope)
        export CURRENT_MODEL_ID="$MODEL_ID"
        export CURRENT_TARGET_DEVICE="$TARGET_DEVICE"
        export CURRENT_LOCAL_NAME="$LOCAL_NAME"
        export CURRENT_TOOL_PARSER="$TOOL_PARSER"
        export CURRENT_REASONING_PARSER="$REASONING_PARSER"
        export CURRENT_OVMS_IMAGE="$OVMS_IMAGE"
        export CURRENT_GPU_DEVICE="$GPU_DEVICE"
        export CURRENT_MODEL_CACHE_DIR="$MODEL_CACHE_DIR"
        export CURRENT_OVMS_REST_PORT="$OVMS_REST_PORT"
        export CURRENT_OVMS_GRPC_PORT="$OVMS_GRPC_PORT"
        export CURRENT_OPENHANDS_PORT="$OPENHANDS_PORT"
        export CURRENT_http_proxy="${http_proxy:-}"
        export CURRENT_https_proxy="${https_proxy:-}"
        export CURRENT_HTTP_PROXY="${HTTP_PROXY:-}"
        export CURRENT_HTTPS_PROXY="${HTTPS_PROXY:-}"
        export CURRENT_no_proxy="${no_proxy:-}"
        export CURRENT_NO_PROXY="${NO_PROXY:-}"

        # Source metadata file (overwrites variables with stored values)
        source "$metadata_file"

        # Check metadata version compatibility
        if [[ "${METADATA_VERSION:-}" != "1" ]]; then
            echo "outdated"
            exit 1
        fi

        # Compare stored values (from source) against current values (CURRENT_ prefix)
        [[ "$MODEL_ID" == "$CURRENT_MODEL_ID" ]] || exit 1
        [[ "$TARGET_DEVICE" == "$CURRENT_TARGET_DEVICE" ]] || exit 1
        [[ "$LOCAL_NAME" == "$CURRENT_LOCAL_NAME" ]] || exit 1
        [[ "$TOOL_PARSER" == "$CURRENT_TOOL_PARSER" ]] || exit 1
        [[ "$REASONING_PARSER" == "$CURRENT_REASONING_PARSER" ]] || exit 1
        [[ "$OVMS_IMAGE" == "$CURRENT_OVMS_IMAGE" ]] || exit 1
        [[ "$GPU_DEVICE" == "$CURRENT_GPU_DEVICE" ]] || exit 1
        [[ "$MODEL_CACHE_DIR" == "$CURRENT_MODEL_CACHE_DIR" ]] || exit 1
        [[ "$OVMS_REST_PORT" == "$CURRENT_OVMS_REST_PORT" ]] || exit 1
        [[ "$OVMS_GRPC_PORT" == "$CURRENT_OVMS_GRPC_PORT" ]] || exit 1
        [[ "$OPENHANDS_PORT" == "$CURRENT_OPENHANDS_PORT" ]] || exit 1
        [[ "${http_proxy:-}" == "$CURRENT_http_proxy" ]] || exit 1
        [[ "${https_proxy:-}" == "$CURRENT_https_proxy" ]] || exit 1
        [[ "${HTTP_PROXY:-}" == "$CURRENT_HTTP_PROXY" ]] || exit 1
        [[ "${HTTPS_PROXY:-}" == "$CURRENT_HTTPS_PROXY" ]] || exit 1
        [[ "${no_proxy:-}" == "$CURRENT_no_proxy" ]] || exit 1
        [[ "${NO_PROXY:-}" == "$CURRENT_NO_PROXY" ]] || exit 1

        echo "ok"
        exit 0
    )

    # Check if metadata is outdated
    if [[ "$result" == "outdated" ]]; then
        METADATA_OUTDATED=1
        return 1
    fi

    # Check if all comparisons succeeded (result is "ok")
    if [[ "$result" != "ok" ]]; then
        return 1
    fi

    return 0
}

# Generate docker-compose.yml from the template
# Uses explicit variable allowlist to avoid unintended substitutions
generate_compose_from_template() {
    local template_file="$1"
    local output_file="$2"

    echo "Generating docker-compose.yml from template..."

    # envsubst requires space-separated variable names, not comma-separated
    # Substitute ALL deployment-managed variables to generate a complete, self-contained
    # runtime compose file. The generated docker-compose.yml contains concrete values,
    # not placeholders.
    envsubst '$MODEL_ID $LOCAL_NAME $TARGET_DEVICE $TOOL_PARSER $REASONING_PARSER $MODEL_CACHE_DIR $HF_TOKEN $OVMS_IMAGE $GPU_DEVICE $WSL_LIBS $OVMS_REST_PORT $OVMS_GRPC_PORT $OPENHANDS_PORT $http_proxy $https_proxy $HTTP_PROXY $HTTPS_PROXY $no_proxy $NO_PROXY' < "$template_file" > "$output_file"
    echo "  Generated: $output_file"
}

################################################################################
# Docker Compose Deployment
################################################################################

deploy_ovms() {
    local compose_file="$1"
    local template_file="$COMPOSE_TEMPLATE"
    local metadata_file="$DEPLOYMENT_METADATA"
    local generation_timestamp
    generation_timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    echo "Deploying OVMS and OpenHands via Docker Compose..."

    # Determine deployment action based on existing state
    if [[ -f "$compose_file" ]]; then
        # Compose file exists - compare deployment fingerprint
        METADATA_OUTDATED=0
        if compare_deployment_fingerprint "$metadata_file"; then
            # Identical deployment fingerprint - preserve existing compose and user edits
            echo "Deployment fingerprint matches - reusing existing docker-compose.yml"
            echo "  (User modifications preserved)"
            # Update timestamp in metadata
            sed -i.bak "s/^GENERATION_TIMESTAMP=.*/GENERATION_TIMESTAMP=${generation_timestamp}/" "$metadata_file"
            rm -f "${metadata_file}.bak"
        else
            # Deployment fingerprint differs - regenerate compose
            if [[ "$METADATA_OUTDATED" -eq 1 ]]; then
                echo "Deployment metadata is outdated or incompatible. Regenerating deployment configuration..."
            else
                echo "Deployment configuration changed - regenerating compose..."
            fi
            docker compose -f "$compose_file" down 2>/dev/null || true
            rm -f "$compose_file"
            generate_compose_from_template "$template_file" "$compose_file"
            write_deployment_metadata "$metadata_file" "$generation_timestamp"
        fi
    else
        # No existing compose - fresh deployment
        echo "No existing docker-compose.yml found."
        echo "Generating from: $template_file"
        generate_compose_from_template "$template_file" "$compose_file"
        write_deployment_metadata "$metadata_file" "$generation_timestamp"
    fi

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
    echo "  export REASONING_PARSER=\"$REASONING_PARSER\""
    echo "  export MODEL_CACHE_DIR=\"$MODEL_CACHE_DIR"
    echo "  export HF_TOKEN=\"\${HF_TOKEN:-}\""
    echo ""
    echo "  # Deploy via Docker Compose"
    echo "  docker compose -f $COMPOSE_FILE up -d"
    echo ""
    echo "  # Wait for OVMS to become ready"
    echo "  curl -sf http://localhost:\${OVMS_REST_PORT:-8000}/v1/config | grep AVAILABLE"
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

    # Resolve reasoning parser if not overridden
    REASONING_PARSER="$(resolve_reasoning_parser "$MODEL_ID" "$REASONING_PARSER")"

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
    echo "  - OpenHands:   http://localhost:${OPENHANDS_PORT}"
    echo ""
    echo "Next steps (from README.md):"
    echo "  1. Verify OVMS: curl http://localhost:${OVMS_REST_PORT}/v3/models"
    echo "  2. Open OpenHands: http://localhost:${OPENHANDS_PORT}"
    echo "  3. Create an agent task to test the integration"
    echo ""
}

# Execute main function with all arguments
main "$@"
