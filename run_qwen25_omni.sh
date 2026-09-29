#!/usr/bin/env bash
set -euo pipefail

cd /ovms

MODEL_REPOSITORY_PATH="${MODEL_REPOSITORY_PATH:-/ovms/models}"
mkdir -p "$MODEL_REPOSITORY_PATH"

BAZEL_STARTUP_ARGS=(--output_user_root=/root/.cache/bazel/bazel)
BAZEL_BUILD_ARGS=(--config=mp_on_py_on)
BAZEL_BIN="$(bazel "${BAZEL_STARTUP_ARGS[@]}" info bazel-bin)"
OVMS_BIN="$BAZEL_BIN/src/ovms"
MEDIAPIPE_RUNTIME="$BAZEL_BIN/src/libovms_mediapipe_runtime_shared.so"

if [[ ! -x "$OVMS_BIN" || ! -f "$MEDIAPIPE_RUNTIME" ]]; then
    echo "Building OVMS and the MediaPipe runtime library..."
    bazel "${BAZEL_STARTUP_ARGS[@]}" build "${BAZEL_BUILD_ARGS[@]}" //src:ovms //src:ovms_mediapipe_runtime_shared
fi

if [[ ! -x "$OVMS_BIN" ]]; then
    echo "OVMS binary not found: $OVMS_BIN" >&2
    exit 1
fi
if [[ ! -f "$MEDIAPIPE_RUNTIME" ]]; then
    echo "MediaPipe runtime library not found: $MEDIAPIPE_RUNTIME" >&2
    exit 1
fi

echo "Using OVMS binary: $OVMS_BIN"
echo "Using MediaPipe runtime: $MEDIAPIPE_RUNTIME"

export OVMS_MEDIA_URL_ALLOW_REDIRECTS=1

"$OVMS_BIN" \
  --rest_port 9112 \
  --model_repository_path "$MODEL_REPOSITORY_PATH" \
  --source_model OpenVINO/qwen3_omni_dense_int4 \
  --log_level TRACE \
    --allowed_media_domains raw.githubusercontent.com
