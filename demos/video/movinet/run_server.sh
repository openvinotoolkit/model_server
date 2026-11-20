#!/usr/bin/env bash
# Run inside the -build container from /ovms.
# Usage: ./demos/video/movinet/run_server.sh SERVER_IP [ALLOWED_ORIGIN]
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 SERVER_IP [ALLOWED_ORIGIN]" >&2
    exit 1
fi

SERVER_IP="$1"
ALLOWED_ORIGIN="${2:-http://localhost:8000}"
REST_PORT="${REST_PORT:-6666}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OVMS_ROOT="${OVMS_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
MODEL_PATH="${MOVINET_MODEL_PATH:-${OVMS_ROOT}/models/OpenVINO/movinet_openvino/movinet_a0_stream.xml}"
LABELS_PATH="${MOVINET_LABELS_PATH:-${OVMS_ROOT}/local/movinet/kinetics_600_labels.txt}"
RUNTIME_DIR="${OVMS_ROOT}/local/movinet/runtime"

cd "${OVMS_ROOT}"

for asset in "${MODEL_PATH}" "${LABELS_PATH}"; do
    if [[ ! -s "${asset}" ]]; then
        echo "Missing MoViNet asset: ${asset}. See demos/video/movinet/README.md." >&2
        exit 1
    fi
done

bazel --output_user_root="${BAZEL_OUTPUT_USER_ROOT:-/root/.cache/bazel/bazel}" \
    build --jobs="${BAZEL_JOBS:-8}" --define BUILD_WITH_WEBRTC=1 //src:ovms

mkdir -p "${RUNTIME_DIR}/graph"
python3 - "${SCRIPT_DIR}" "${RUNTIME_DIR}" "${MODEL_PATH}" "${LABELS_PATH}" <<'PY'
import json
import sys
from pathlib import Path

demo_dir, runtime_dir, model_path, labels_path = map(Path, sys.argv[1:])
graph = (demo_dir / "graph.pbtxt").read_text()
graph = graph.replace(
    'model_path: "/ovms/models/OpenVINO/movinet_openvino/movinet_a0_stream.xml"',
    f"model_path: {json.dumps(str(model_path.resolve()))}",
).replace(
    'labels_path: "/ovms/local/movinet/kinetics_600_labels.txt"',
    f"labels_path: {json.dumps(str(labels_path.resolve()))}",
)
(runtime_dir / "graph" / "graph.pbtxt").write_text(graph)
config = json.loads((demo_dir / "config.json").read_text())
config["mediapipe_config_list"][0]["base_path"] = str(runtime_dir / "graph")
(runtime_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
PY

OVMS_WEBRTC_ANNOUNCED_IP="${SERVER_IP}" \
OVMS_WEBRTC_VIDEO_GRAPH=movinet \
exec ./bazel-bin/src/ovms --rest_port "${REST_PORT}" --port 0 --rest_workers 2 \
    --config_path "${RUNTIME_DIR}/config.json" \
    --allowed_origins "${ALLOWED_ORIGIN}"
