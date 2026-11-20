#!/usr/bin/env bash
set -euo pipefail

usage() {
        cat <<'HELP'
Usage:
    bash tests/docker_compose_smoke_test.sh <full-image> <component|all> [options]

Components:
    base      (default image, +0 MB)
    gpu       (+500 MB)       npu       (+300 MB)
    opencv    (+50 MB)        python   (+95 MB)
    tensorflow (+5 MB)        paddle    (+2 MB)
    onnx       (+5 MB)        espeak   (+10 MB)
    all        (run every component)

Component sizes are approximate increases over the base image, measured from
openvino/model_server-gpu:rasapala on Ubuntu 24.04. Other source images and
driver versions can change the results.

Options:
    --skip-pull       Use the local full image without pulling it.
    --with-python     Also retain Python in a non-Python component image.
    --with-opencv     Also retain OpenCV libraries in a non-OpenCV component image.
    --report PATH     Write the Markdown run report to PATH.
    --demo            Run Level 2 using repository fixtures.
    --demo-fixtures DIR  Run Level 2 using a custom fixture directory.
    -h, --help        Show this help and exit.

The default report is test_log/docker_compose_smoke_report.md in the repository.
Each component builds a composed image, checks its requested artifact, starts
the local dummy IR model, waits at most 60 seconds for readiness, then gives
the KServe v2 inference request 15 seconds. The all mode runs components
sequentially, records every result, and exits nonzero if any component fails.
The GPU case needs a host render device and runs inference on GPU. NPU checks
artifact retention and the common CPU inference path; it does not exercise NPU
inference because that requires NPU hardware and a component-specific model.
HELP
}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} == -h || ${1:-} == --help ]]; then
        usage
        exit 0
fi

if [[ $# -lt 2 ]]; then
    usage
    exit 2
fi

FULL_IMAGE=$1
COMPONENT=$2
shift 2
SKIP_PULL=0
WITH_PYTHON=0
WITH_OPENCV=0
REPORT_FILE="$ROOT_DIR/test_log/docker_compose_smoke_report.md"
DEMO_FIXTURES=
RUN_OPTIONS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-pull)
            SKIP_PULL=1
            RUN_OPTIONS+=(--skip-pull)
            ;;
        --with-python)
            WITH_PYTHON=1
            RUN_OPTIONS+=(--with-python)
            ;;
        --with-opencv)
            WITH_OPENCV=1
            RUN_OPTIONS+=(--with-opencv)
            ;;
        --report)
            if [[ $# -lt 2 || -z $2 ]]; then
                echo "--report requires a path" >&2
                exit 2
            fi
            REPORT_FILE=$2
            shift
            ;;
        --demo)
            DEMO_FIXTURES="$ROOT_DIR/tests/docker_compose_fixtures"
            RUN_OPTIONS+=(--demo)
            ;;
        --demo-fixtures)
            if [[ $# -lt 2 || -z $2 || $2 == --* ]]; then
                echo "--demo-fixtures requires a directory" >&2
                exit 2
            fi
            DEMO_FIXTURES=$2
            RUN_OPTIONS+=(--demo-fixtures "$DEMO_FIXTURES")
            shift
            ;;
        *) usage; exit 2 ;;
    esac
    shift
done

case "$COMPONENT" in
    all|base|gpu|npu|opencv|python|tensorflow|paddle|onnx|espeak) ;;
    *) usage; exit 2 ;;
esac

if [[ ${OVMS_COMPOSER_SMOKE_SINGLE:-0} != 1 ]]; then
    if [[ $COMPONENT == all ]]; then
        COMPONENTS=(base gpu npu opencv python tensorflow paddle onnx espeak)
    else
        COMPONENTS=("$COMPONENT")
    fi

    mkdir -p "$(dirname "$REPORT_FILE")"
    {
        echo "# Docker Composer Smoke-Test Report"
        echo
        echo "- Full image: \`$FULL_IMAGE\`"
        echo "- Run started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "- Components: \`${COMPONENTS[*]}\`"
        if [[ -n $DEMO_FIXTURES ]]; then
            echo "- Level 2 fixtures: \`$DEMO_FIXTURES\`"
        fi
        echo
        echo "| Component | Result | Summary |"
        echo "| --- | --- | --- |"
    } > "$REPORT_FILE"

    FAILED=0
    for component in "${COMPONENTS[@]}"; do
        LOG_FILE=$(mktemp)
        echo "Running composer smoke test: $component"
        if OVMS_COMPOSER_SMOKE_SINGLE=1 bash "$0" "$FULL_IMAGE" "$component" "${RUN_OPTIONS[@]}" >"$LOG_FILE" 2>&1; then
            cat "$LOG_FILE"
            case "$component" in
                base) detail="Default image; dummy IR model and bounded KServe v2 inference passed." ;;
                gpu) detail="GPU artifact retained; dummy IR inference targeted GPU and passed." ;;
                npu) detail="NPU artifacts retained; dummy IR inference passed. NPU execution was not exercised." ;;
                opencv) detail="OpenCV libraries retained; dummy IR model and KServe v2 inference passed." ;;
                python) detail="Python artifacts retained; dummy IR model and KServe v2 inference passed." ;;
                tensorflow|paddle|onnx) detail="$component frontend artifact retained; dummy IR model and KServe v2 inference passed." ;;
                espeak) detail="eSpeak library/data retained; dummy IR model and KServe v2 inference passed." ;;
            esac
            printf '| `%s` | PASS | %s |\n' "$component" "$detail" >> "$REPORT_FILE"
        else
            cat "$LOG_FILE" >&2
            if grep -q '^PASS: .*completed KServe v2 inference' "$LOG_FILE"; then
                printf '| `%s` | PASS | Level 1: artifact and dummy IR checks passed. |\n' "$component" >> "$REPORT_FILE"
            fi
            FAILURE=$(grep '^error: Level 2 ' "$LOG_FILE" | tail -n 1 || true)
            if [[ -z $FAILURE ]]; then
                FAILURE=$(grep -Ei 'error:|failed|missing|timed out|requires a host' "$LOG_FILE" | tail -n 1 || true)
            fi
            if [[ -z $FAILURE ]]; then
                FAILURE="Failed; see command output for details."
            fi
            FAILURE=${FAILURE//\|/\/}
            if grep -q '^Level 2:' "$LOG_FILE"; then
                printf '| `%s` Level 2 | FAIL | %s |\n' "$component" "$FAILURE" >> "$REPORT_FILE"
            else
                printf '| `%s` | FAIL | %s |\n' "$component" "$FAILURE" >> "$REPORT_FILE"
            fi
            FAILED=1
        fi
        if grep -q '^PASS: Level 2 ' "$LOG_FILE"; then
            printf '| `%s` Level 2 | PASS | %s |\n' "$component" "$(sed -n 's/^PASS: Level 2 //p' "$LOG_FILE")" >> "$REPORT_FILE"
        fi
        rm -f "$LOG_FILE"
    done

    echo >> "$REPORT_FILE"
    echo "Report written to $REPORT_FILE"
    exit "$FAILED"
fi

WORK_DIR=$(mktemp -d)
IMAGE_NAME="ovms-compose-smoke:${COMPONENT}"
CONTAINER_NAME="ovms-compose-smoke-${COMPONENT}-$$"
EXPECTED_PATHS=()
COMPOSER_FLAGS=()
DOCKER_DEVICE_ARGS=()
OVMS_ARGS=()

case "$COMPONENT" in
    base) ;;
    gpu)
        COMPOSER_FLAGS+=(--gpu)
        EXPECTED_PATHS+=("/ovms/lib/libopenvino_intel_gpu_plugin.so*")
        RENDER_NODE=$(find /dev/dri -maxdepth 1 -name 'renderD*' -print -quit 2>/dev/null || true)
        if [[ -z $RENDER_NODE ]]; then
            echo "GPU smoke test requires a host render device under /dev/dri" >&2
            exit 2
        fi
        DOCKER_DEVICE_ARGS+=(--device /dev/dri --group-add "$(stat -c '%g' "$RENDER_NODE")")
        OVMS_ARGS+=(--target_device GPU)
        ;;
    npu)
        COMPOSER_FLAGS+=(--npu)
        EXPECTED_PATHS+=("/ovms/lib/libopenvino_intel_npu_plugin.so*" "/ovms/lib/libopenvino_intel_npu_compiler.so*")
        ;;
    opencv)
        COMPOSER_FLAGS+=(--opencv)
        EXPECTED_PATHS+=("/ovms/lib/libopencv*.so*")
        ;;
    python)
        COMPOSER_FLAGS+=(--python)
        EXPECTED_PATHS+=("/ovms/lib/libovmspython.so*")
        ;;
    tensorflow)
        COMPOSER_FLAGS+=(--tensorflow)
        EXPECTED_PATHS+=("/ovms/lib/libopenvino_tensorflow_frontend.so*")
        ;;
    paddle)
        COMPOSER_FLAGS+=(--paddle)
        EXPECTED_PATHS+=("/ovms/lib/libopenvino_paddle_frontend.so*")
        ;;
    onnx)
        COMPOSER_FLAGS+=(--onnx)
        EXPECTED_PATHS+=("/ovms/lib/libopenvino_onnx_frontend.so*")
        ;;
    espeak)
        COMPOSER_FLAGS+=(--espeak)
        EXPECTED_PATHS+=("/ovms/lib/libespeak-ng.so*" "/ovms/share/espeak-ng-data/phontab")
        ;;
    *) usage; exit 2 ;;
esac

if [[ $WITH_PYTHON -eq 1 && $COMPONENT != python ]]; then
    COMPOSER_FLAGS+=(--python)
    EXPECTED_PATHS+=("/ovms/lib/libovmspython.so*")
fi
if [[ $WITH_OPENCV -eq 1 && $COMPONENT != opencv ]]; then
    COMPOSER_FLAGS+=(--opencv)
    EXPECTED_PATHS+=("/ovms/lib/libopencv*.so*")
fi

if [[ $SKIP_PULL -eq 1 ]]; then
    COMPOSER_FLAGS+=(--skip-pull)
fi

cleanup() {
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

echo "Building composed image $IMAGE_NAME from $FULL_IMAGE ($COMPONENT)"
python3 "$ROOT_DIR/docker_compose.py" \
    --full-image "$FULL_IMAGE" \
    --output-name "$IMAGE_NAME" \
    --work-dir "$WORK_DIR" \
    "${COMPOSER_FLAGS[@]}"

MODEL_DIR="$ROOT_DIR/src/test/dummy/1"
docker run -d --name "$CONTAINER_NAME" \
    "${DOCKER_DEVICE_ARGS[@]}" \
    -p 127.0.0.1::9000 \
    -v "$MODEL_DIR:/models:ro" \
    "$IMAGE_NAME" \
    --model_path /models/dummy.xml --model_name dummy --rest_port 9000 \
    "${OVMS_ARGS[@]}" >/dev/null

if [[ $(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true) != true ]]; then
    echo "OVMS container exited before component artifacts could be checked" >&2
    docker logs "$CONTAINER_NAME" >&2 || true
    exit 1
fi

for path in "${EXPECTED_PATHS[@]}"; do
    if ! docker exec "$CONTAINER_NAME" bash -lc 'compgen -G "$1" >/dev/null' _ "$path"; then
        echo "Expected component artifact is missing: $path" >&2
        docker logs "$CONTAINER_NAME" >&2 || true
        exit 1
    fi
done

HOST_PORT=$(docker port "$CONTAINER_NAME" 9000/tcp | sed 's/.*://')
MODEL_READY_URL="http://127.0.0.1:${HOST_PORT}/v2/models/dummy/ready"
DEADLINE=$((SECONDS + 60))
until curl --silent --fail --max-time 2 "$MODEL_READY_URL" >/dev/null 2>&1; do
    if [[ $(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true) != true ]]; then
        echo "OVMS container exited before the dummy model became ready" >&2
        docker logs "$CONTAINER_NAME" >&2 || true
        exit 1
    fi
    if [[ $SECONDS -ge $DEADLINE ]]; then
        echo "Timed out waiting for dummy model readiness: $MODEL_READY_URL" >&2
        docker logs "$CONTAINER_NAME" >&2 || true
        exit 1
    fi
    sleep 1
done

INFERENCE_URL="http://127.0.0.1:${HOST_PORT}/v2/models/dummy/infer"
if ! curl --silent --show-error --fail --max-time 15 \
    -H 'Content-Type: application/json' \
    -d '{"inputs":[{"name":"b","shape":[1,10],"datatype":"FP32","data":[0,1,2,3,4,5,6,7,8,9]}]}' \
    "$INFERENCE_URL" -o "$WORK_DIR/inference.json"; then
    echo "Inference request failed: $INFERENCE_URL" >&2
    docker logs "$CONTAINER_NAME" >&2 || true
    exit 1
fi

if ! grep -q '"outputs"' "$WORK_DIR/inference.json"; then
    echo "Inference response did not contain outputs:" >&2
    cat "$WORK_DIR/inference.json" >&2
    docker logs "$CONTAINER_NAME" >&2 || true
    exit 1
fi

echo "PASS: $COMPONENT image contains its expected artifacts and completed KServe v2 inference"
cat "$WORK_DIR/inference.json"
echo

if [[ -n $DEMO_FIXTURES && $COMPONENT != base ]]; then
    echo "Level 2: $COMPONENT"
    docker rm -f "$CONTAINER_NAME" >/dev/null
    bash "$ROOT_DIR/tests/docker_compose_demo_probe.sh" "$COMPONENT" \
        "$DEMO_FIXTURES" "$IMAGE_NAME" "$CONTAINER_NAME" "$WORK_DIR" \
        "${DOCKER_DEVICE_ARGS[@]}"
fi