#!/usr/bin/env bash
# Run inside the OVMS build container with the checkout mounted at /ovms.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 SERVER_IP [ALLOWED_ORIGIN]" >&2
    echo "Example: $0 127.0.0.1" >&2
    exit 2
fi

SERVER_IP="$1"
ALLOWED_ORIGIN="${2:-http://localhost:8000}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OVMS_ROOT="${OVMS_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
MODEL_NAME="yolox_tiny-fp16-ov"
MODEL_REPO="OpenVINO/${MODEL_NAME}"
MODEL_DIR="${OVMS_MODEL_DIR:-${OVMS_ROOT}/models/yolo_bytetrack}"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"
LABELS_PATH="${MODEL_DIR}/coco_80cl.txt"
MODEL_XML="${MODEL_PATH}/1/${MODEL_NAME}.xml"
MODEL_BIN="${MODEL_PATH}/1/${MODEL_NAME}.bin"
RUNTIME_DIR="${MODEL_DIR}/runtime"
PYTHON="${PYTHON:-python3}"
BAZEL="${BAZEL:-bazel}"

cd "${OVMS_ROOT}"
"${BAZEL}" \
    --output_user_root="${BAZEL_OUTPUT_USER_ROOT:-/root/.cache/bazel/bazel}" \
    build --jobs="${BAZEL_JOBS:-8}" --define BUILD_WITH_WEBRTC=1 //src:ovms

mkdir -p "${MODEL_DIR}"
if [[ ! -s "${MODEL_XML}" || ! -s "${MODEL_BIN}" || ! -s "${LABELS_PATH}" ]]; then
    DOWNLOAD_VENV="${MODEL_DIR}/.venv"
    "${PYTHON}" -m venv --system-site-packages --without-pip "${DOWNLOAD_VENV}"
    DOWNLOAD_PYTHON="${DOWNLOAD_VENV}/bin/python"
    if ! "${DOWNLOAD_PYTHON}" -c 'import openvino; from huggingface_hub import snapshot_download'; then
        echo "Installing model-download dependencies in ${DOWNLOAD_VENV}"
        "${PYTHON}" -m pip --python "${DOWNLOAD_PYTHON}" install openvino huggingface_hub
    fi
    echo "Downloading and preparing ${MODEL_REPO} under ${MODEL_DIR}"
    (
        cd "${MODEL_DIR}"
        "${DOWNLOAD_PYTHON}" "${OVMS_ROOT}/demos/mediapipe/bytetrack/download_models.py" \
            --model-repo "${MODEL_REPO}"
    )
fi

mkdir -p "${RUNTIME_DIR}"
"${PYTHON}" - \
    "${SCRIPT_DIR}/graph.pbtxt" \
    "${SCRIPT_DIR}/config.json" \
    "${RUNTIME_DIR}" \
    "${MODEL_PATH}" \
    "${LABELS_PATH}" \
    "${MODEL_NAME}" <<'PY'
import json
import sys
from pathlib import Path

graph_path, config_path, runtime_dir, model_path, labels_path, model_name = sys.argv[1:]
runtime_dir = Path(runtime_dir)
graph_dir = runtime_dir / "graph"
graph_dir.mkdir(exist_ok=True)
graph = Path(graph_path).read_text()

servable = 'servable_name: "yolox_tiny-fp16-ov"'
label_map = 'label_map_path: "/demo/coco_80cl.txt"'
if graph.count(servable) != 1 or graph.count(label_map) != 1:
    raise SystemExit("Expected YOLOX servable and label-map entries were not found in graph.pbtxt")
graph = graph.replace(servable, f"servable_name: {json.dumps(model_name)}")
graph = graph.replace(label_map, f"label_map_path: {json.dumps(labels_path)}")
(graph_dir / "graph.pbtxt").write_text(graph)

config = json.loads(Path(config_path).read_text())
config["model_config_list"][0]["config"]["name"] = model_name
config["model_config_list"][0]["config"]["base_path"] = model_path
graph_config = config["mediapipe_config_list"][0]
graph_config["base_path"] = str(graph_dir)
graph_config["graph_path"] = "graph.pbtxt"
(runtime_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
PY

export OVMS_WEBRTC_ANNOUNCED_IP="${SERVER_IP}"
export OVMS_WEBRTC_VIDEO_GRAPH="YoloByteTrackWebRTC"
export OVMS_WEBRTC_PORT_RANGE="${OVMS_WEBRTC_PORT_RANGE:-52000-52100}"

exec "${OVMS_ROOT}/bazel-bin/src/ovms" \
    --config_path "${RUNTIME_DIR}/config.json" \
    --rest_port "${REST_PORT:-6666}" \
    --port 0 \
    --rest_workers "${REST_WORKERS:-2}" \
    --allowed_origins "${ALLOWED_ORIGIN}"
