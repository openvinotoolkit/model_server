#!/usr/bin/env bash
cd /ovms

export OVMS_WEBRTC_OMNI_MODEL_PATH="${OVMS_WEBRTC_OMNI_MODEL_PATH:-/ovms/models/OpenVINO/qwen3_omni_dense_int4}"
export OVMS_WEBRTC_STT_MODEL_PATH="${OVMS_WEBRTC_STT_MODEL_PATH:-/ovms/src/test/llm_testing/openai/whisper-tiny}"
export OVMS_WEBRTC_STT_DEVICE="${OVMS_WEBRTC_STT_DEVICE:-CPU}"
export OVMS_WEBRTC_OMNI_DEVICE="${OVMS_WEBRTC_OMNI_DEVICE:-CPU}"
export OVMS_WEBRTC_TALKER_DEVICE="${OVMS_WEBRTC_TALKER_DEVICE:-CPU}"

bazel --output_user_root=/root/.cache/bazel/bazel build //src:ovms
OVMS_BIN="$(bazel --output_user_root=/root/.cache/bazel/bazel info bazel-bin)/src/ovms"

"$OVMS_BIN" \
  --rest_port 9112 \
  --config_path /ovms/config.json
