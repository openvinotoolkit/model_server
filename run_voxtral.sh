#!/usr/bin/env bash
cd /ovms


#bazel --output_user_root=/root/.cache/bazel/bazel build //src:ovms
OVMS_BIN="$(bazel --output_user_root=/root/.cache/bazel/bazel info bazel-bin)/src/ovms"

 #--config_path /ovms/config.json \
"$OVMS_BIN" \
  --rest_port 9112 \
  --log_level TRACE \
  --source_model OpenVINO/voxtral-realtime-2602 \
  --model_repository_path /ovms/models \
  --model_name voxtral-asr \
  --task speech2text \
  --target_device CPU
