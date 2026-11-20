#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Container lifecycle and Level 2 requests for the standalone composer smoke runner."""

import json
import math
import re
import struct
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

if __package__:
    from .utils.ports import probe_port
else:
    from utils.ports import probe_port

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/docker_compose_fixtures"
MODEL_DIRS = {
    "tensorflow": "src/test/dummy_tflite",
    "onnx": "demos/using_onnx_model/python/workspace/resnet50-onnx",
    "paddle": "demos/classification_using_paddlepaddle_model/python/model",
    "espeak": "src/test/llm_testing/hexgrad/Kokoro-82M",
    "gpu": "src/test/llm_testing/thenlper/gte-small/ov",
    "npu": "src/test/llm_testing/thenlper/gte-small/ov",
}


def docker(*args, check=True, timeout=15):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"docker {args[0]} failed: {result.stderr.strip() or result.stdout.strip()}")
    return result


def mount(source, destination):
    return ["-v", f"{source}:{destination}:ro"]


def request(url, timeout, body=None, header_length=None):
    command = ["curl", "--silent", "--show-error", "--fail-with-body", "--noproxy", "*", "--max-time", str(timeout)]
    if body is not None:
        if header_length is None:
            command += ["-H", "Content-Type: application/json"]
        else:
            command += ["-H", "Content-Type: application/octet-stream",
                        "-H", f"Inference-Header-Content-Length: {header_length}"]
        command += ["--data-binary", "@-"]
    result = subprocess.run([*command, url], input=body, capture_output=True, timeout=timeout + 2)
    if result.returncode:
        error = result.stderr.decode(errors="replace").strip()
        detail = result.stdout[:1024].decode(errors="replace").strip()
        raise RuntimeError(f"{url}: {error} {detail}".strip())
    return result.stdout


@contextmanager
def server(container, image, docker_args, server_args):
    host = "127.0.0.1"
    port = probe_port(0, host)
    if port is None:
        raise RuntimeError("could not find a free loopback port for OVMS")
    try:
        docker("run", "-d", "--name", container, "-p", f"{host}:{port}:9000", *docker_args,
               image, *server_args, "--rest_port", "9000", "--rest_workers", "2", timeout=30)
        yield f"http://{host}:{port}"
    except Exception:
        try:
            logs = docker("logs", "--tail", "40", container, check=False)
            print(logs.stdout + logs.stderr, end="", flush=True)
        except (OSError, subprocess.SubprocessError):
            pass
        raise
    finally:
        try:
            docker("rm", "-f", container, check=False)
        except (OSError, subprocess.SubprocessError) as error:
            print(f"Could not clean up {container}: {error}", flush=True)


def require_running(container):
    if docker("inspect", "-f", "{{.State.Running}}", container, check=False).stdout.strip() != "true":
        raise RuntimeError("OVMS container exited before the model became ready")


def wait_ready(container, url, model):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            request(f"{url}/v2/models/{model}/ready", 2)
            return
        except (RuntimeError, subprocess.TimeoutExpired):
            require_running(container)
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for {model} model readiness")


def validate_wav(payload):
    if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError("not a WAV")
    offset = 12
    frame_size = audio_size = 0
    while offset + 8 <= len(payload):
        chunk, size = struct.unpack_from("<4sI", payload, offset)
        offset += 8
        if offset + size > len(payload):
            raise ValueError("truncated WAV chunk")
        if chunk == b"fmt ":
            if size < 16:
                raise ValueError("invalid WAV format chunk")
            encoding, _, _, _, frame_size, _ = struct.unpack_from("<HHIIHH", payload, offset)
            if encoding not in (1, 3) or frame_size <= 0:
                raise ValueError("unsupported WAV format")
        elif chunk == b"data":
            audio_size = size
        offset += size + (size % 2)
    if not frame_size or audio_size < frame_size or audio_size % frame_size:
        raise ValueError("missing WAV samples")


def validate_response(component, payload, repo_fixture):
    if component == "espeak":
        validate_wav(payload)
        return
    result = json.loads(payload)
    if component in ("gpu", "npu"):
        if not result["data"][0]["embedding"]:
            raise ValueError("empty embedding")
        return
    outputs = result["outputs"]
    if not outputs or not all(output.get("data") for output in outputs):
        raise ValueError("empty inference output")
    if component == "opencv":
        data = outputs[0]["data"]
        if len(data) != 60 or set(data) != {11, 21, 31}:
            raise ValueError("incorrect decoded pixels")
    elif component == "python" and repo_fixture:
        (output,) = outputs
        expected = {"name": "out", "datatype": "FP32", "shape": [1, 3], "data": [11, 22, 33]}
        if any(output.get(key) != value for key, value in expected.items()):
            raise ValueError(f"incorrect tensor from Python node: {outputs}")


def model_mounts(component, work_dir):
    if component == "python":
        handler = "src/test/mediapipe/python/scripts/add_two_inputs.py"
        return (mount(ROOT / "src/test/mediapipe/graphpythonsummator.pbtxt", "/summator.pbtxt")
                + mount(ROOT / handler, f"/ovms/{handler}"))
    model_dir = ROOT / MODEL_DIRS[component]
    if not model_dir.is_dir():
        raise RuntimeError(f"missing model directory {model_dir}; see composer.md for one-time setup")
    args = mount(model_dir, "/model")
    if component in ("gpu", "npu"):
        graph = (model_dir / "graph.pbtxt").read_text()
        graph = re.sub(r'models_path:.*', 'models_path: "/model"', graph)
        graph = re.sub(r'target_device:.*', f'target_device: "{component.upper()}"', graph)
        path = work_dir / "embeddings.pbtxt"
        path.write_text(graph)
        args += mount(path, "/embeddings.pbtxt")
    return args


def demo_request(component, fixture, endpoint, repo_fixture):
    if component == "opencv":
        image = b"P6\n1 1\n255\n\x0a\x14\x1e"
        body = struct.pack("<I", len(image)) + image
        header = json.dumps({"inputs": [{"name": "input", "shape": [1], "datatype": "BYTES",
                                         "parameters": {"binary_data_size": len(body)}}]}).encode()
        return header + body, len(header)
    if repo_fixture and component in ("onnx", "paddle"):
        return None, None
    binary = fixture / "request.bin"
    if endpoint.startswith("/v2/") and binary.is_file():
        length = (fixture / "header-length").read_text().strip()
        if not re.fullmatch(r"[1-9][0-9]*", length):
            raise ValueError("header-length must be a positive integer")
        return binary.read_bytes(), int(length)
    return (fixture / "request.json").read_bytes(), None


def run_demo(component, fixtures, image, container, work_dir, device_args):
    endpoint = {"gpu": "/v1/embeddings", "npu": "/v1/embeddings",
                "espeak": "/v1/audio/speech"}.get(component, "/v2/models/smoke/infer")
    fixture = (fixtures / component).resolve()
    repo_fixture = fixture == FIXTURES / component
    args = list(device_args)
    if component == "npu":
        device = Path("/dev/accel/accel0")
        if not device.is_char_device():
            raise RuntimeError(f"missing host NPU device {device}")
        args += ["--device", "/dev/accel", "--group-add", str(device.stat().st_gid)]
    if component == "opencv":
        args += mount(ROOT / "src/test/increment_1x3x4x5", "/ir")
        server_args = ["--model_name", "smoke", "--model_path", "/ir", "--layout", "NHWC:NCHW"]
    else:
        config = fixture / "config.json"
        if not config.is_file():
            raise RuntimeError(f"missing {config}; see composer.md for fixture setup")
        args += mount(fixture, "/demo")
        if repo_fixture:
            args += model_mounts(component, work_dir)
        server_args = ["--config_path", "/demo/config.json"]
    body, header_length = demo_request(component, fixture, endpoint, repo_fixture)
    with server(container, image, args, server_args) as url:
        wait_ready(container, url, "smoke")
        if body is None:
            (tensor,) = json.loads(request(f"{url}/v2/models/smoke", 5))["inputs"]
            shape = tensor["shape"]
            if len(shape) != 4 or min(shape) <= 0 or math.prod(shape) > 224 * 224 * 3:
                raise ValueError(f"unexpected classifier input shape: {shape}")
            tensor["data"] = [0] * math.prod(shape)
            body = json.dumps({"inputs": [tensor]}).encode()
        validate_response(component, request(url + endpoint, 30, body, header_length), repo_fixture)
        if component == "opencv":
            logs = docker("logs", container)
            if "Using OpenCV image decoder:" not in logs.stdout + logs.stderr:
                raise RuntimeError("OpenCV decoder was not used")
    return f"{endpoint} returned the expected response."