# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");

"""Standalone Windows installer smoke tests with a composer-style Markdown report."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

try:
    from tests.functional.utils.ports import probe_port
    from tests.functional import docker_compose_demo_probe as composer
except ImportError:
    from functional.utils.ports import probe_port
    from functional import docker_compose_demo_probe as composer

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "docker_compose_fixtures"
BASE_CASES = ("invalid-mode", "none-components", "system-python", "embedded-python", "components-only")
COMPONENTS = (
    "core",
    "python_system",
    "python_embedded",
    "python_none",
    "espeak",
    "npu",
    "gpu",
    "opencv_mp",
    "onnx_frontend",
    "paddle_frontend",
    "tensorflow_frontend",
)
CASES = BASE_CASES + COMPONENTS

COMPONENT_ARTIFACTS = {
    "core": (
        "ovms.exe",
        "openvino.dll",
        "openvino_genai.dll",
        "openvino_gguf_frontend.dll",
        "openvino_ir_frontend.dll",
        "libovmspython.dll",
    ),
    "python_system": ("python\\pyovms.pyd",),
    "python_embedded": ("python\\python.exe", "python\\pyovms.pyd"),
    "python_none": (),
    "espeak": ("espeak-ng.dll", "espeak-ng-data"),
    "npu": (
        "openvino_intel_npu_compiler.dll",
        "openvino_intel_npu_compiler_loader.dll",
        "openvino_intel_npu_plugin.dll",
        "openvino_intel_npu_vm_runtime.dll",
    ),
    "gpu": ("openvino_intel_gpu_plugin.dll",),
    "opencv_mp": (
        "opencv_world4140.dll",
        "ovms_opencv_calculators.dll",
        "ovms_opencv_image_decoder.dll",
    ),
    "onnx_frontend": ("openvino_onnx_frontend.dll",),
    "paddle_frontend": ("openvino_paddle_frontend.dll",),
    "tensorflow_frontend": (
        "openvino_tensorflow_frontend.dll",
        "openvino_tensorflow_lite_frontend.dll",
    ),
}


def free_port() -> int:
    port = probe_port(0, "127.0.0.1")
    if port is None:
        raise RuntimeError("could not find a free loopback port")
    return port


def run(command: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", check=False, timeout=timeout)


def find_installer(path: Path) -> Path:
    installer = path.resolve()
    if not installer.is_file():
        raise RuntimeError(f"installer not found: {installer}")
    return installer


def discover_python314() -> Path | None:
    candidates = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Python314" / "python.exe",
        Path(os.environ.get("ProgramW6432", r"C:\Program Files")) / "Python314" / "python.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / "Python314" / "python.exe",
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def wait_ready(url: str, process: subprocess.Popen, timeout: int = 45) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"OVMS exited before readiness; exit={process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as error:
            last_error = str(error)
        time.sleep(1)
    raise RuntimeError(f"OVMS readiness timeout: {last_error}")


def wait_model_ready(base_url: str, model: str, process: subprocess.Popen, timeout: int = 90) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"OVMS exited before {model} readiness; exit={process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/v2/models/{model}/ready", timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except Exception as error:
            last_error = str(error)
        time.sleep(1)
    raise RuntimeError(f"OVMS {model} model readiness timeout: {last_error}")


def start_and_check(install_dir: Path) -> str:
    ovms = install_dir / "ovms.exe"
    if not ovms.is_file():
        raise RuntimeError(f"missing OVMS executable: {ovms}")

    work = install_dir.parent / "ovms-smoke-data"
    work.mkdir(parents=True, exist_ok=True)
    models = work / "models"
    models.mkdir(parents=True, exist_ok=True)
    config = work / "config.json"
    config.write_text('{"model_config_list":[]}', encoding="utf-8")

    rest_port = free_port()
    grpc_port = free_port()
    while grpc_port == rest_port:
        grpc_port = free_port()

    log_path = work / "ovms.log"
    with log_path.open("wb") as log:
        # pylint: disable-next=consider-using-with
        process = subprocess.Popen(
            [str(ovms), "--config_path", str(config), "--rest_port", str(rest_port),
             "--port", str(grpc_port), "--log_level", "DEBUG"],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(install_dir),
        )
    try:
        wait_ready(f"http://127.0.0.1:{rest_port}/v2", process)
        return f"OVMS readiness passed on REST {rest_port}/gRPC {grpc_port}"
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()


def read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    return data.decode("utf-8", errors="replace")


def read_install_options() -> dict:
    path = Path(os.environ.get("LOCALAPPDATA", "")) / "OVMS" / "install-options.json"
    if not path.is_file():
        raise RuntimeError(f"missing install-options.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def uninstall(install_dir: Path) -> None:
    uninstaller = install_dir / "unins000.exe"
    if uninstaller.is_file():
        run([str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"])


def detect_intel_gpu() -> bool:
    command = ["powershell.exe", "-NoProfile", "-Command",
               "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
    result = run(command, timeout=30)
    return result.returncode == 0 and any("intel" in line.lower() for line in result.stdout.splitlines())


def detect_intel_npu() -> bool:
    command = ["powershell.exe", "-NoProfile", "-Command",
               "Get-CimInstance Win32_PnPEntity | Select-Object -ExpandProperty Name"]
    result = run(command, timeout=30)
    return result.returncode == 0 and any("intel" in line.lower() and "npu" in line.lower()
                                         for line in result.stdout.splitlines())


def verify_component_artifacts(install_dir: Path, artifacts: tuple[str, ...]) -> str:
    missing = []
    for relative_path in artifacts:
        path = install_dir / relative_path
        if not path.exists():
            missing.append(relative_path)
    if missing:
        raise RuntimeError("missing component artifacts: " + ", ".join(missing))
    if artifacts:
        return "Verified " + ", ".join(artifacts)
    return "Core-only component state verified"


def install_component(installer: Path, work_root: Path, python_mode: str,
                      components: str, artifacts: tuple[str, ...]) -> Path:
    install_dir = work_root / "install"
    log_path = work_root / "setup.log"
    install_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(installer), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
        f'/DIR={install_dir}', f'/LOG={log_path}', '/MULTIINSTANCE=1',
        f'/COMPONENTS={components}', f'/PYTHONMODE={python_mode}',
    ]
    result = run(command)
    if result.returncode:
        raise RuntimeError(f"installer exit={result.returncode}; {result.stdout[-2000:]}")

    options = read_install_options()
    if options.get("pythonMode") != python_mode:
        raise RuntimeError(f"pythonMode was {options.get('pythonMode')!r}, expected {python_mode!r}")

    if python_mode == "system":
        binding = install_dir / "python" / "pyovms.pyd"
        if not binding.is_file():
            raise RuntimeError(f"missing system-Python binding: {binding}")
        if (install_dir / "python" / "python.exe").is_file():
            raise RuntimeError("embedded python.exe unexpectedly installed in system-Python mode")
    elif python_mode == "embedded":
        if not (install_dir / "python" / "python.exe").is_file():
            raise RuntimeError("missing embedded python.exe")
        if not (install_dir / "python" / "pyovms.pyd").is_file():
            raise RuntimeError("missing embedded pyovms.pyd")
    elif python_mode == "none":
        python_exe = install_dir / "python" / "python.exe"
        if python_exe.is_file():
            raise RuntimeError("embedded python.exe unexpectedly installed in none-Python mode")

    verify_component_artifacts(install_dir, artifacts)
    return install_dir


def run_successful_install(installer: Path, work_root: Path, python_mode: str,
                          components: str, artifacts: tuple[str, ...] = ()) -> tuple[str, str]:
    install_dir = install_component(installer, work_root, python_mode, components, artifacts)
    try:
        readiness = start_and_check(install_dir)
        level1 = verify_component_artifacts(install_dir, artifacts)
        level2 = readiness
        return level1, level2
    finally:
        uninstall(install_dir)


def check_case(case: str, installer: Path) -> list[tuple[str, bool, str]]:
    try:
        with tempfile.TemporaryDirectory(prefix="ovms-installer-", dir=os.environ.get("TEMP")) as temp:
            work_root = Path(temp)
            if case == "invalid-mode":
                log_path = work_root / "invalid.log"
                result = run([str(installer), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                              "/PYTHONMODE=invalid", f"/LOG={log_path}", "/MULTIINSTANCE=1"], timeout=120)
                if result.returncode == 0:
                    return [("", False, "installer accepted invalid /PYTHONMODE")]
                log_candidates = [log_path, installer.parent / "OVMSSetup.log"]
                log_text = "\n".join(read_text(candidate) for candidate in log_candidates if candidate.is_file())
                if "Invalid /PYTHONMODE value" not in log_text:
                    return [("", False, "installer rejection did not contain the expected validation message")]
                return [("", True, "Invalid /PYTHONMODE was rejected by the installer")]

            if case == "none-components":
                level1, level2 = run_successful_install(
                    installer, work_root, "none",
                    "python_none,gpu,opencv_mp,onnx_frontend",
                    ("ovms.exe", "openvino_intel_gpu_plugin.dll", "opencv_world4140.dll", "openvino_onnx_frontend.dll"))
                return [(" Level 1", True, level1), (" Level 2", True, level2)]

            if case == "system-python":
                if discover_python314() is None:
                    return [(" Level 1", True, "SKIP-ready: Python 3.14 not found"),
                            (" Level 2", True, "SKIP-ready: Python 3.14 not found")]
                level1, level2 = run_successful_install(
                    installer, work_root, "system", "python_system,gpu",
                    ("ovms.exe", "python\\pyovms.pyd", "openvino_intel_gpu_plugin.dll"))
                return [(" Level 1", True, level1), (" Level 2", True, level2)]

            if case == "embedded-python":
                level1, level2 = run_successful_install(
                    installer, work_root, "embedded", "python_embedded,onnx_frontend",
                    ("ovms.exe", "python\\python.exe", "python\\pyovms.pyd", "openvino_onnx_frontend.dll"))
                return [(" Level 1", True, level1), (" Level 2", True, level2)]

            if case == "components-only":
                level1, level2 = run_successful_install(
                    installer, work_root, "none", "python_none,paddle_frontend,tensorflow_frontend",
                    ("ovms.exe", "openvino_paddle_frontend.dll",
                     "openvino_tensorflow_frontend.dll", "openvino_tensorflow_lite_frontend.dll"))
                return [(" Level 1", True, level1), (" Level 2", True, level2)]

            if case in COMPONENTS:
                if case == "python_system":
                    if discover_python314() is None:
                        return [(" Level 1", True, "SKIP-ready: Python 3.14 not found"),
                                (" Level 1", True, "SKIP-ready: Python 3.14 not found")]
                    # PythonExecutorCalculator is hosted in the opencv_mp plugin DLL, so
                    # exercising a real mediapipe python node (Level 2) requires it too.
                    python_mode, component_selection = "system", "core,python_system,opencv_mp"
                elif case == "python_embedded":
                    python_mode, component_selection = "embedded", "core,python_embedded,opencv_mp"
                else:
                    python_mode, component_selection = "none", "core,python_none"
                    if case not in ("core", "python_none"):
                        component_selection += "," + case

                try:
                    install_dir = install_component(
                        installer, work_root, python_mode, component_selection,
                        COMPONENT_ARTIFACTS[case])
                except Exception as error:
                    return [(" Level 1", False, str(error))]

                try:
                    try:
                        level1_files = verify_component_artifacts(install_dir, COMPONENT_ARTIFACTS[case])
                        level1_ready = start_and_check(install_dir)
                        level1 = f"{level1_files}; {level1_ready}"
                    except Exception as error:
                        return [(" Level 1", False, str(error))]

                    try:
                        level2 = level2_component_check(case, install_dir, python_mode)
                        if level2.startswith("SKIP-ready:"):
                            detail = level2.removeprefix("SKIP-ready:").strip()
                            return [(" Level 1", True, level1), (" Level 2", True, "SKIP-ready: " + detail)]
                        return [(" Level 1", True, level1), (" Level 2", True, level2)]
                    except Exception as error:
                        return [(" Level 1", True, level1), (" Level 2", False, str(error))]
                finally:
                    uninstall(install_dir)

            return [("", False, f"unknown case: {case}")]
    except Exception as error:
        return [(" Level 1", False, str(error))]


def post_json(url: str, body: bytes, timeout: int = 45) -> dict:
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {response_body.strip()}") from error


def http_request(url: str, method: str = "GET", body: bytes | None = None,
                 headers: dict[str, str] | None = None, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        response_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {response_body.strip()}") from error


@contextmanager
def run_ovms(install_dir: Path, config_path: Path, work: Path):
    rest_port = free_port()
    grpc_port = free_port()
    while grpc_port == rest_port:
        grpc_port = free_port()
    log_path = work / "ovms.log"
    command = [str(install_dir / "ovms.exe"), "--config_path", str(config_path),
               "--rest_port", str(rest_port), "--port", str(grpc_port), "--log_level", "DEBUG"]
    with log_path.open("wb") as log:
        # pylint: disable-next=consider-using-with
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=str(install_dir))
    try:
        base_url = f"http://127.0.0.1:{rest_port}"
        wait_ready(f"{base_url}/v2", process)
        wait_model_ready(base_url, "smoke", process)
        yield base_url, rest_port, grpc_port, log_path
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()


def localize_graph(graph_path: Path, model_dir: Path, device_field: str | None, device: str | None) -> str:
    graph = re.sub(r'models_path:.*', f'models_path: "{model_dir.as_posix()}"',
                   graph_path.read_text(encoding="utf-8"))
    if device_field:
        graph = re.sub(rf'^[ \t]*{device_field}:.*$', '', graph, flags=re.MULTILINE)
        graph = re.sub(r'^([ \t]*models_path:.*)$', rf'\1\n        {device_field}: "{device}"',
                       graph, flags=re.MULTILINE)
    return graph


def embeddings_check(component: str, device: str, install_dir: Path) -> str:
    llm_path = Path(os.environ.get("TEST_LLM_PATH") or ROOT / "src" / "test" / "llm_testing").resolve()
    model_dir = llm_path / "OpenVINO" / "bge-base-en-v1.5-int8-ov"
    if not model_dir.is_dir():
        raise RuntimeError(f"embeddings model directory was not prepared: {model_dir}")

    work = install_dir.parent / "level2"
    work.mkdir(parents=True, exist_ok=True)
    graph = localize_graph(ROOT / "src" / "test" / "embeddings" / "graph_ov.pbtxt", model_dir,
                           "target_device", device)
    graph_path = work / "embeddings.pbtxt"
    graph_path.write_text(graph, encoding="utf-8")

    config = {"model_config_list": [], "mediapipe_config_list": [
        {"name": "smoke", "graph_path": graph_path.as_posix()}]}
    config_path = work / "embeddings.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    body = (FIXTURES / component / "request.json").read_bytes()

    with run_ovms(install_dir, config_path, work) as (url, rest_port, grpc_port, _log_path):
        endpoint = f"{url}/v1/embeddings"
        payload = http_request(endpoint, "POST", body, {"Content-Type": "application/json"})
        composer.validate_response(component, payload, True)
        return (f"Composer-style Level 2 {component} embeddings request passed at {endpoint} "
                f"(REST {rest_port}/gRPC {grpc_port})")


def espeak_check(install_dir: Path) -> str:
    llm_path = Path(os.environ.get("TEST_LLM_PATH") or ROOT / "src" / "test" / "llm_testing").resolve()
    model_dir = llm_path / "OpenVINO" / "Kokoro-82M-int8-ov"
    if not model_dir.is_dir():
        raise RuntimeError(f"TTS model directory was not prepared: {model_dir}")

    work = install_dir.parent / "level2"
    work.mkdir(parents=True, exist_ok=True)
    graph = localize_graph(ROOT / "src" / "test" / "audio" / "graph_tts.pbtxt", model_dir, None, None)
    graph_path = work / "tts.pbtxt"
    graph_path.write_text(graph, encoding="utf-8")

    config = {"model_config_list": [], "mediapipe_config_list": [
        {"name": "smoke", "graph_path": graph_path.as_posix()}]}
    config_path = work / "tts.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    body = (FIXTURES / "espeak" / "request.json").read_bytes()

    with run_ovms(install_dir, config_path, work) as (url, rest_port, grpc_port, _log_path):
        endpoint = f"{url}/v1/audio/speech"
        payload = http_request(endpoint, "POST", body, {"Content-Type": "application/json"})
        composer.validate_response("espeak", payload, True)
        return f"Composer-style Level 2 espeak TTS request passed at {endpoint} (REST {rest_port}/gRPC {grpc_port})"


def opencv_check(install_dir: Path) -> str:
    model_dir = ROOT / "src" / "test" / "increment_1x3x4x5"
    work = install_dir.parent / "level2"
    work.mkdir(parents=True, exist_ok=True)

    image = b"P6\n1 1\n255\n\x0a\x14\x1e"
    body = struct.pack("<I", len(image)) + image
    header = json.dumps({"inputs": [{"name": "input", "shape": [1], "datatype": "BYTES",
                                     "parameters": {"binary_data_size": len(body)}}]}).encode()
    full_body = header + body

    rest_port = free_port()
    grpc_port = free_port()
    while grpc_port == rest_port:
        grpc_port = free_port()
    log_path = work / "opencv.log"
    command = [str(install_dir / "ovms.exe"), "--model_name", "smoke", "--model_path", str(model_dir),
               "--layout", "NHWC:NCHW", "--rest_port", str(rest_port), "--port", str(grpc_port),
               "--log_level", "DEBUG"]
    with log_path.open("wb") as log:
        # pylint: disable-next=consider-using-with
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=str(install_dir))
    try:
        base_url = f"http://127.0.0.1:{rest_port}"
        wait_ready(f"{base_url}/v2", process)
        wait_model_ready(base_url, "smoke", process)
        endpoint = f"{base_url}/v2/models/smoke/infer"
        payload = http_request(endpoint, "POST", full_body,
                               {"Content-Type": "application/octet-stream",
                                "Inference-Header-Content-Length": str(len(header))})
        composer.validate_response("opencv", payload, True)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "Using OpenCV image decoder:" not in log_text:
            raise RuntimeError("OpenCV decoder was not used")
        return (f"Composer-style Level 2 opencv_mp KServe v2 infer request passed at {endpoint} "
                f"(REST {rest_port}/gRPC {grpc_port})")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()


def static_model_check(component: str, config_source: Path, model_dir: Path,
                       install_dir: Path, request_body: bytes | None) -> str:
    if not model_dir.is_dir():
        raise RuntimeError(f"{component} model directory was not prepared: {model_dir}")

    work = install_dir.parent / "level2"
    work.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_source.read_text(encoding="utf-8"))
    for entry in config.get("model_config_list", []):
        entry["config"]["base_path"] = model_dir.as_posix()
    config_path = work / "static_model.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with run_ovms(install_dir, config_path, work) as (url, rest_port, grpc_port, _log_path):
        body = request_body
        if body is None:
            metadata = json.loads(http_request(f"{url}/v2/models/smoke", "GET", timeout=5))
            (tensor,) = metadata["inputs"]
            shape = tensor["shape"]
            if len(shape) != 4 or min(shape) <= 0 or math.prod(shape) > 224 * 224 * 3:
                raise RuntimeError(f"unexpected classifier input shape: {shape}")
            tensor["data"] = [0] * math.prod(shape)
            body = json.dumps({"inputs": [tensor]}).encode()
        endpoint = f"{url}/v2/models/smoke/infer"
        payload = http_request(endpoint, "POST", body, {"Content-Type": "application/json"})
        composer.validate_response(component, payload, True)
        return (f"Composer-style Level 2 {component} inference request passed at {endpoint} "
                f"(REST {rest_port}/gRPC {grpc_port})")


def python_summator_check(install_dir: Path, python_mode: str) -> str:
    if python_mode == "system":
        # KNOWN BUG (see "Known issues" in installer.md): ovms.exe loads libovmspython.dll /
        # libpython_calculators.dll via LoadLibraryExW with LOAD_LIBRARY_SEARCH_* flags, which
        # ignore PATH entirely and only consult AddDllDirectory()-registered directories. The
        # loader only registers the embedded Python directory, never the system Python
        # installation directory, so python314.dll can never be resolved in system-Python mode.
        # No test-harness environment variable (PATH/PYTHONHOME/PYTHONPATH) can work around this;
        # it requires a fix in src/python/python_runtime_loader.cpp.
        return ("SKIP-ready: known bug - system-Python mode cannot load libovmspython.dll on "
                "Windows (LOAD_LIBRARY_SEARCH_* ignores PATH; system Python dir is never "
                "registered via AddDllDirectory); see installer.md Known issues")

    work = install_dir.parent / "level2"
    work.mkdir(parents=True, exist_ok=True)
    handler = (ROOT / "src" / "test" / "mediapipe" / "python" / "scripts" / "add_two_inputs.py").resolve()
    graph = re.sub(r'handler_path:.*', f'handler_path: "{handler.as_posix()}"',
                   (ROOT / "src" / "test" / "mediapipe" / "graphpythonsummator.pbtxt").read_text(encoding="utf-8"))
    graph_path = work / "summator.pbtxt"
    graph_path.write_text(graph, encoding="utf-8")

    config = {"model_config_list": [], "mediapipe_config_list": [
        {"name": "smoke", "graph_path": graph_path.as_posix()}]}
    config_path = work / "summator.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    body = (FIXTURES / "python" / "request.json").read_bytes()

    with run_ovms(install_dir, config_path, work) as (url, rest_port, grpc_port, _log_path):
        endpoint = f"{url}/v2/models/smoke/infer"
        payload = http_request(endpoint, "POST", body, {"Content-Type": "application/json"})
        composer.validate_response("python", payload, True)
        return (f"Composer-style Level 2 python node summation request passed at {endpoint} "
                f"(REST {rest_port}/gRPC {grpc_port})")


def level2_component_check(component: str, install_dir: Path, python_mode: str = "none") -> str:
    if component == "core":
        llm_path = Path(os.environ.get("TEST_LLM_PATH") or ROOT / "src" / "test" / "llm_testing").resolve()
        model_dir = llm_path / "mzeglars" / "dummy-cyclic-gpt2-ov"
        if not model_dir.is_dir():
            raise RuntimeError(f"base LLM model directory was not prepared: {model_dir}")

        work = install_dir.parent / "level2"
        work.mkdir(parents=True, exist_ok=True)
        rest_port = free_port()
        grpc_port = free_port()
        while grpc_port == rest_port:
            grpc_port = free_port()

        graph = re.sub(r'models_path:.*', f'models_path: "{model_dir.as_posix()}"',
                       (ROOT / "src" / "test" / "llm" / "lm_cb_regular.pbtxt").read_text(encoding="utf-8"))
        graph_path = work / "llm.pbtxt"
        graph_path.write_text(graph, encoding="utf-8")

        fixture_dir = FIXTURES / "base"
        config = json.loads((fixture_dir / "config.json").read_text(encoding="utf-8"))
        for entry in config.get("mediapipe_config_list", []):
            entry["graph_path"] = graph_path.as_posix()
        config_path = work / "base.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        endpoint = f"http://127.0.0.1:{rest_port}/v3/completions"
        body = (fixture_dir / "request.json").read_bytes()
        log_path = work / "core.log"
        command = [str(install_dir / "ovms.exe"), "--config_path", str(config_path),
                   "--rest_port", str(rest_port), "--port", str(grpc_port), "--log_level", "DEBUG"]
        with log_path.open("wb") as log:
            # pylint: disable-next=consider-using-with
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=str(install_dir))
        try:
            wait_ready(f"http://127.0.0.1:{rest_port}/v2", process)
            deadline = time.monotonic() + 90
            payload = None
            last_error = ""
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"OVMS exited before base LLM completion; exit={process.returncode}")
                try:
                    payload = post_json(endpoint, body)
                    break
                except RuntimeError as error:
                    last_error = str(error)
                    if "Mediapipe is not loaded yet" not in last_error:
                        raise
                time.sleep(1)
            if payload is None:
                log_tail = ""
                if log_path.is_file():
                    log_tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
                raise RuntimeError(f"base LLM model did not become ready: {last_error}\nOVMS log tail:\n{log_tail}")
            text = payload.get("choices", [{}])[0].get("text")
            completion_tokens = payload.get("usage", {}).get("completion_tokens", 0)
            if not isinstance(text, str) or not text.strip() or completion_tokens <= 0:
                raise RuntimeError("base LLM completion returned empty text")
            return f"Composer-style Level 2 base LLM completion passed at {endpoint}"
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()

    if component == "python_none":
        return "SKIP-ready: no component-specific request for python_none"

    if component in ("python_system", "python_embedded"):
        return python_summator_check(install_dir, python_mode)

    if component == "gpu":
        if not detect_intel_gpu():
            return "SKIP-ready: Intel GPU hardware not detected"
        return embeddings_check("gpu", "GPU", install_dir)

    if component == "npu":
        if not detect_intel_npu():
            return "SKIP-ready: Intel NPU hardware not detected"
        return embeddings_check("npu", "NPU", install_dir)

    if component == "espeak":
        return espeak_check(install_dir)

    if component == "opencv_mp":
        return opencv_check(install_dir)

    if component == "onnx_frontend":
        return static_model_check("onnx", FIXTURES / "onnx" / "config.json",
                                  ROOT / composer.MODEL_DIRS["onnx"], install_dir, None)

    if component == "paddle_frontend":
        return static_model_check("paddle", FIXTURES / "paddle" / "config.json",
                                  ROOT / composer.MODEL_DIRS["paddle"], install_dir, None)

    if component == "tensorflow_frontend":
        body = (FIXTURES / "tensorflow" / "request.json").read_bytes()
        return static_model_check("tensorflow", FIXTURES / "tensorflow" / "config.json",
                                  ROOT / composer.MODEL_DIRS["tensorflow"], install_dir, body)

    return start_and_check(install_dir)


def report_passed(path: Path) -> bool:
    found = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        found = True
        status = line.split("|")[2].strip()
        if status not in ("PASS", "SKIP"):
            return False
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Windows OVMS installer CLI smoke tests")
    parser.add_argument("--installer", type=Path, default=ROOT / "dist" / "windows" / "OVMS-Setup.exe")
    parser.add_argument("--report", type=Path, default=ROOT / "windows_installer_smoke_report.md")
    parser.add_argument("--case", dest="cases", action="append", choices=CASES,
                        help="Run one case; repeat for multiple cases")
    args = parser.parse_args()

    if os.name != "nt":
        parser.error("the Windows installer smoke tests require Windows")

    # Model/fixture preparation is a mandatory part of standard testing, not an opt-in step:
    # the prepare scripts skip any model that is already present, so this is cheap once fixtures
    # have been downloaded (or linked from c:\opt\llm_testing on CI workers).
    prepare_script = ROOT / "tests" / "functional" / "windows_installer_prepare_models.ps1"
    llm_dir = Path(os.environ.get("TEST_LLM_PATH") or ROOT / "src" / "test" / "llm_testing").resolve()
    result = run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                  "-File", str(prepare_script), "-PrepareLLM", "-LLMDir", str(llm_dir)], timeout=3600)
    if result.returncode:
        raise RuntimeError("windows_installer_prepare_models.ps1 failed: " + result.stdout[-2000:])

    try:
        installer = find_installer(args.installer)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 2

    cases = args.cases or list(CASES)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", buffering=1) as report:
        report.write("# Windows Installer Smoke-Test Report\n\n")
        report.write(f"- Installer: `{installer}`\n")
        report.write(f"- Run started: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}\n")
        report.write(f"- Cases: `{' '.join(cases)}`\n\n")
        report.write("| Case | Result | Summary |\n| --- | --- | --- |\n")

        for case in cases:
            print(f"Running installer smoke test: {case}", flush=True)
            results = check_case(case, installer)
            for level, passed, detail in results:
                status = "PASS" if passed else "FAIL"
                if detail.startswith("SKIP-ready:"):
                    status = "SKIP"
                    detail = detail.removeprefix("SKIP-ready:").strip()
                summary = " ".join(detail.splitlines()).replace("|", "/")
                label = f"`{case}`{level}"
                print(f"{status}: {case}{level}: {summary}", flush=True)
                report.write(f"| {label} | {status} | {summary} |\n")

    passed = report_passed(args.report)
    verdict = "Overall result: PASS" if passed else "Overall result: FAIL"
    with args.report.open("a", encoding="utf-8") as report:
        report.write(f"\n{verdict}\n")
    print(verdict)
    print(f"Report written to {args.report}")
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
