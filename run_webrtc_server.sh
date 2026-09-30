#!/usr/bin/env bash
cd /ovms

export OVMS_WEBRTC_OMNI_MODEL_PATH="${OVMS_WEBRTC_OMNI_MODEL_PATH:-/ovms/models/OpenVINO/qwen3_omni_dense_int4}"

#bazel --output_user_root=/root/.cache/bazel/bazel build //src:ovms
OVMS_BIN="$(bazel --output_user_root=/root/.cache/bazel/bazel info bazel-bin)/src/ovms"
echo $OVMS_BIN
ls -la $OVMS_BIN

"$OVMS_BIN" \
  --rest_port 9112 \
  --log_level INFO \
  --config_path /ovms/config.json
