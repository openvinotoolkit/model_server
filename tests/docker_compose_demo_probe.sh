#!/usr/bin/env bash
set -euo pipefail

component=$1
fixtures=$2
image=$3
container=$4
work_dir=$5
shift 5
device_args=("$@")
fixture="$(realpath -m -- "$fixtures/$component")"
model=smoke
server_args=()
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

fail() {
    echo "error: Level 2 $component: $*" >&2
    docker logs --tail 40 "$container" >&2 2>/dev/null || true
    exit 1
}

case "$component" in
    gpu|npu)
        endpoint=/v1/embeddings
        if [[ $component == npu ]]; then
            [[ -c /dev/accel/accel0 ]] || fail 'missing host NPU device /dev/accel/accel0'
            device_args+=(--device /dev/accel --group-add "$(stat -c '%g' /dev/accel/accel0)")
        fi
        ;;
    espeak) endpoint=/v1/audio/speech ;;
    *) endpoint=/v2/models/smoke/infer ;;
esac

repo_fixture=false
if [[ $fixture == "$root/tests/docker_compose_fixtures/$component" ]]; then
    repo_fixture=true
    case "$component" in
        tensorflow) model_dir="$root/src/test/dummy_tflite" ;;
        onnx) model_dir="$root/demos/using_onnx_model/python/workspace/resnet50-onnx" ;;
        paddle) model_dir="$root/demos/classification_using_paddlepaddle_model/python/model" ;;
        python)
            device_args+=(-v "$root/src/test/mediapipe/graphpythonsummator.pbtxt:/summator.pbtxt:ro"
                -v "$root/src/test/mediapipe/python/scripts/add_two_inputs.py:/ovms/src/test/mediapipe/python/scripts/add_two_inputs.py:ro")
            ;;
        espeak) model_dir="$root/src/test/llm_testing/hexgrad/Kokoro-82M" ;;
        gpu|npu) model_dir="$root/src/test/llm_testing/thenlper/gte-small/ov" ;;
    esac
    if [[ -n ${model_dir:-} ]]; then
        [[ -d $model_dir ]] || fail "missing model directory $model_dir; see composer.md for one-time setup"
        device_args+=(-v "$model_dir:/model:ro")
    fi
    if [[ $component == gpu || $component == npu ]]; then
        [[ -r $model_dir/graph.pbtxt ]] || fail "missing $model_dir/graph.pbtxt; see composer.md for one-time setup"
        sed -E "s@models_path:.*@models_path: \"/model\"@; s@target_device:.*@target_device: \"${component^^}\"@" \
            "$model_dir/graph.pbtxt" > "$work_dir/embeddings.pbtxt"
        device_args+=(-v "$work_dir/embeddings.pbtxt:/embeddings.pbtxt:ro")
    fi
fi

if [[ $component == opencv ]]; then
    fixture="$work_dir/opencv"
    mkdir "$fixture"
    device_args+=(-v "$root/src/test/increment_1x3x4x5:/ir:ro")
    server_args=(--model_name smoke --model_path /ir --layout NHWC:NCHW)
    python3 - "$fixture" <<'PY'
import json
import pathlib
import struct
import sys

fixture = pathlib.Path(sys.argv[1])
image = b"P6\n1 1\n255\n\x0a\x14\x1e"
body = struct.pack("<I", len(image)) + image
header = json.dumps({"inputs": [{"name": "input", "shape": [1], "datatype": "BYTES",
                                "parameters": {"binary_data_size": len(body)}}]}).encode()
(fixture / "request.bin").write_bytes(header + body)
(fixture / "header-length").write_text(str(len(header)))
PY
fi

if [[ ${#server_args[@]} -eq 0 ]]; then
    [[ -r $fixture/config.json ]] || fail "missing $fixture/config.json; see composer.md for fixture setup"
    server_args=(--config_path /demo/config.json)
fi
headers=(-H 'Content-Type: application/json')
request="$fixture/request.json"
if [[ -f $fixture/request.bin && $endpoint == /v2/* ]]; then
    [[ -r $fixture/header-length ]] || fail "missing $fixture/header-length"
    header_length=$(<"$fixture/header-length")
    [[ $header_length =~ ^[1-9][0-9]*$ ]] || fail 'header-length must be a positive integer'
    headers=(-H 'Content-Type: application/octet-stream' -H "Inference-Header-Content-Length: $header_length")
    request="$fixture/request.bin"
fi
if [[ $repo_fixture != true || ( $component != onnx && $component != paddle ) ]]; then
    [[ -r $request ]] || fail "missing $request; see composer.md for fixture setup"
fi

docker run -d --name "$container" "${device_args[@]}" -p 127.0.0.1::9000 \
    -v "$fixture:/demo:ro" "$image" "${server_args[@]}" --rest_port 9000 >/dev/null || fail 'container startup failed'
port=$(docker port "$container" 9000/tcp) || fail 'cannot find published REST port'
url="http://${port}"
deadline=$((SECONDS + 60))
until curl --silent --fail --max-time 2 "$url/v2/models/$model/ready" >/dev/null; do
    [[ $(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null) == true ]] || fail 'server exited before demo was ready'
    (( SECONDS < deadline )) || fail 'timed out waiting for demo model readiness'
    sleep 1
done

if [[ $repo_fixture == true && ( $component == onnx || $component == paddle ) ]]; then
    request="$work_dir/classifier-request.json"
    curl --silent --show-error --fail --max-time 5 "$url/v2/models/$model" -o "$work_dir/metadata.json" || fail 'cannot read model inputs'
    python3 - "$work_dir/metadata.json" "$request" <<'PY' || fail 'cannot construct classifier request'
import json
import math
import pathlib
import sys

metadata, request = map(pathlib.Path, sys.argv[1:])
(tensor,) = json.loads(metadata.read_text())["inputs"]
assert len(tensor["shape"]) == 4 and 0 < math.prod(tensor["shape"]) <= 224 * 224 * 3
tensor["data"] = [0] * math.prod(tensor["shape"])
request.write_text(json.dumps({"inputs": [tensor]}))
PY
fi

response="$work_dir/demo-response"
if ! curl --silent --show-error --fail-with-body --max-time 30 "${headers[@]}" \
    --data-binary "@$request" "$url$endpoint" -o "$response"; then
    if [[ -f $response ]]; then head -c 1024 "$response" >&2; echo >&2; fi
    fail "request to $endpoint failed"
fi

if ! python3 - "$component" "$response" "$repo_fixture" <<'PY'
import json
import pathlib
import struct
import sys

component, path, repo_fixture = sys.argv[1:]
payload = pathlib.Path(path).read_bytes()
if component == "espeak":
    assert payload[:4] == b"RIFF" and payload[8:12] == b"WAVE", "not a WAV"
    offset = 12
    frame_size = 0
    audio_size = 0
    while offset + 8 <= len(payload):
        chunk, size = struct.unpack_from("<4sI", payload, offset)
        offset += 8
        assert offset + size <= len(payload), "truncated WAV chunk"
        if chunk == b"fmt ":
            assert size >= 16, "invalid WAV format chunk"
            encoding, _, _, _, frame_size, _ = struct.unpack_from("<HHIIHH", payload, offset)
            assert encoding in (1, 3) and frame_size > 0, "unsupported WAV format"
        elif chunk == b"data":
            audio_size = size
        offset += size + (size % 2)
    assert frame_size and audio_size >= frame_size and audio_size % frame_size == 0, "missing WAV samples"
else:
    result = json.loads(payload)
    if component in ("gpu", "npu"):
        assert result["data"][0]["embedding"], "empty embedding"
    elif component == "opencv":
        data = result["outputs"][0]["data"]
        assert len(data) == 60 and set(data) == {11, 21, 31}, "incorrect decoded pixels"
    elif component == "python" and repo_fixture == "true":
        (output,) = result["outputs"]
        assert output["name"] == "out" and output["datatype"] == "FP32" and output["shape"] == [1, 3], output
        assert output["data"] == [11, 22, 33], "incorrect sum from Python node"
    else:
        assert result["outputs"] and all(output.get("data") for output in result["outputs"]), "empty inference output"
PY
then
    fail "invalid response from $endpoint"
fi
if [[ $component == opencv ]]; then
    logs=$(docker logs "$container" 2>&1) || fail 'cannot read decoder logs'
    [[ $logs == *'Using OpenCV image decoder:'* ]] || fail 'OpenCV decoder was not used'
fi
echo "PASS: Level 2 $endpoint returned the expected response type."