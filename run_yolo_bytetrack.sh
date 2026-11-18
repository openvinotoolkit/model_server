#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "Usage: $0 (no arguments; use environment variables to override defaults)" >&2
    exit 2
fi

OVMS_ROOT="${OVMS_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
SERVER_IP="${OVMS_WEBRTC_ANNOUNCED_IP:-10.123.247.3}"
ALLOWED_ORIGIN="${ALLOWED_ORIGIN:-http://localhost:8000}"

exec "${OVMS_ROOT}/demos/video/yolo_bytetrack/run_server.sh" \
    "${SERVER_IP}" "${ALLOWED_ORIGIN}"
