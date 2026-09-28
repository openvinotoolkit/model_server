#!/usr/bin/env bash
cd /ovms

#bazel --output_user_root=/root/.cache/bazel/bazel build //src:ovms
OVMS_BIN="$(bazel --output_user_root=/root/.cache/bazel/bazel info bazel-bin)/src/ovms"
echo $OVMS_BIN
ls -la $OVMS_BIN

"$OVMS_BIN" \
  --rest_port 9112 \
  --log_level TRACE \
  --config_path /ovms/config.json
